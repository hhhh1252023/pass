# NPU CI 执行监控
**生成时间**: 2026-08-17 09:21 UTC
**分析 Run 数**: 47

---

## 📊 本次执行总结

- **成功 Job 数**: 201
- **失败 Run 数**: 47
- **成功 Job 平均耗时**: 24.2min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.1min | #31991908670 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906208) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.7min | #31997280258 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293863087) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.6min | #31993937733 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239535) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.5min | #31994038554 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544402) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.3min | #31994368156 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370348) |
| base-b-test-16-npu-a3 / run (0) | 64.3min | #31993851373 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056358) |
| base-b-test-16-npu-a3 / run (0) | 59.8min | #31997280258 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862791) |
| base-b-test-16-npu-a3 / run (0) | 58.0min | #31993937733 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239164) |
| base-b-test-16-npu-a3 / run (0) | 52.7min | #31997517074 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802842) |
| multimodal-gen-test-2-npu-a3 | 51.0min | #29412383315 | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995708) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 45 | 0 | 27 | 0 | 18 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 28 | 5 | 0 | 17 | 6 |
| base-b-test-16-npu-a3 / run (0) | 28 | 10 | 0 | 12 | 6 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 28 | 10 | 0 | 12 | 6 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 28 | 10 | 0 | 12 | 6 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 12 | 0 | 0 | 12 | 0 |
| base-b-test-8-npu-a3 / run (0) | 28 | 11 | 0 | 11 | 6 |
| base-b-test-4-npu-a3 / run (1) | 28 | 11 | 0 | 11 | 6 |
| base-b-test-4-npu-a3 / run (0) | 28 | 11 | 0 | 11 | 6 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 28 | 12 | 0 | 10 | 6 |
| base-b-test-1-npu-a3 / run (0) | 28 | 13 | 0 | 9 | 6 |
| base-b-test-2-npu-a3 / run (0) | 28 | 13 | 0 | 9 | 6 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 10 | 0 | 0 | 9 | 1 |
| stage-b-test-4-npu-a3 | 19 | 0 | 7 | 0 | 12 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 5 | 0 | 0 | 5 | 0 |
| base-a-test-1-npu-a2 / run (0) | 28 | 19 | 0 | 4 | 5 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 10 | 6 | 0 | 4 | 0 |
| stage-b-test-2-npu-a2 (0) | 19 | 14 | 3 | 0 | 2 |
| stage-b-test-1-npu-a2 (0) | 19 | 9 | 3 | 0 | 7 |
| stage-b-test-1-npu-a2 (1) | 19 | 12 | 1 | 0 | 6 |
| multimodal-gen-test-2-npu-a3 | 18 | 6 | 1 | 0 | 11 |

---

## 📋 各用例失败统计

| 用例名称 | 执行次数 | 成功 | 失败 |
|----------|----------|------|------|
| `sglang/multimodal_gen/test/server/ascend/test_server_1_npu.py::TestDiffusionServerOneNpu::test_diffusion_generation[glm_image_t2i_1npu] - Failed: Diffusion testcase 'glm_image_t2i_1npu' failed 1 check(s):` | 6 | 0 | 6 |
| `sglang/multimodal_gen/test/server/ascend/test_server_2_npu.py::TestDiffusionServerTwoNpu::test_diffusion_generation[wan2_2_t2v_14b_w8a8_2npu] - Failed: Diffusion testcase 'wan2_2_t2v_14b_w8a8_2npu' failed 1 check(s):` | 1 | 0 | 1 |

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #29333278063 | `mamba_hicache_fix` | 190.0min | 2 | multimodal-gen-test-1-npu-a3, stage-b-test-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29333278063) |
| #31994353942<br>[#30345 [Intel][XPU][LoRA] Enable LoRA on Intel XPU](https://github.com/sgl-project/sglang/pull/30345) | `enable-lora-xpu` | 185.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31994353942) |
| #29412383315<br>[#31225 Remove dead MiniMax M3 artifacts](https://github.com/sgl-project/sglang/pull/31225) | `agent/minimax-m3-dead-code` | 177.0min | 2 | multimodal-gen-test-1-npu-a3, stage-b-test-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29412383315) |
| #29311302659<br>[#31107 [NPU] Determine the topk norm_type through scoring_func](https://github.com/sgl-project/sglang/pull/31107) | `topk-glm` | 168.5min | 3 | multimodal-gen-test-2-npu-a3, multimodal-gen-test-1-npu-a3, stage-b-test-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29311302659) |
| #29304163296<br>[#31109 Remove QServe and FBGEMM FP8 quantization](https://github.com/sgl-project/sglang/pull/31109) | `remove-qserve-quantization` | 165.9min | 2 | multimodal-gen-test-1-npu-a3, stage-b-test-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29304163296) |
| #31994368156<br>[#33726 fix(bcg): preserve Qwen3-VL DeepStack inputs during replay](https://github.com/sgl-project/sglang/pull/33726) | `fix/bcg-deepstack-replay-slot` | 161.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31994368156) |
| #31994038554<br>[#35004 [Diffusion] Reuse SRT CLIP encoder blocks](https://github.com/sgl-project/sglang/pull/35004) | `codex/diffusion-reuse-srt-clip` | 156.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31994038554) |
| #31993937733<br>[#34926 Clean deprecated DeepSeek V4 Environs](https://github.com/sgl-project/sglang/pull/34926) | `clean-dsv4` | 144.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31993937733) |
| #29319025946<br>[#31110 [CPU] bypass scoring_func argument in topk for cpu device](https://github.com/sgl-project/sglang/pull/31110) | `fix_topk_interface_change` | 136.2min | 5 | stage-b-test-4-npu-a3, multimodal-gen-test-1-npu-a3, stage-b-test-2-npu-a2 (0), stage-b-test-1-npu-a2 (0), stage-b-test-2-npu-a2 (1) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29319025946) |
| #31991908670<br>[#32514 feat(kv-events): Add component_types field to BlockStored for per-component placement tracking](https://github.com/sgl-project/sglang/pull/32514) | `feat/kv-events-component-placement-v2` | 132.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31991908670) |
| #29317020495 | `repo-cache-dtype->fp32` | 127.7min | 5 | stage-b-test-1-npu-a2 (0), stage-b-test-4-npu-a3, stage-b-test-2-npu-a2 (1), stage-b-test-2-npu-a2 (0), multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29317020495) |
| #31997280258<br>[#34938 perf: overlap Qwen shared expert with DeepEP routed experts](https://github.com/sgl-project/sglang/pull/34938) | `yangminl/agentx-decode-gap-v2-shared-overlap-v2-20260815` | 126.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31997280258) |
| #29306153312 | `fix/amd-ci-perf-bounds-and-dispatcher-test` | 119.3min | 2 | multimodal-gen-test-1-npu-a3, stage-b-test-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29306153312) |
| #31999768419<br>[#30805 [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103](https://github.com/sgl-project/sglang/pull/30805) | `dsv4_fp8_trtllm_gen` | 116.3min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31999768419) |
| #29318370591<br>[#28428 [Intel GPU] DeepSeek V4 12/N: use sgl-kernel implementation of silu_and_mul_clamp to run on XPU](https://github.com/sgl-project/sglang/pull/28428) | `ds_v4_xpu_silu_and_mul_clamp` | 115.4min | 4 | stage-b-test-1-npu-a2 (1), stage-b-test-2-npu-a2 (0), stage-b-test-4-npu-a3, stage-b-test-2-npu-a2 (1) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29318370591) |
| #31995334699 | `cheng/gc-s12-carrier` | 113.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31995334699) |
| #31997824508<br>[#30565 [AMD] [GLM5] Fix MTP layer_quant_config in-place mutation + add nextn Quark-exclude unit test](https://github.com/sgl-project/sglang/pull/30565) | `tmp/eagle-mtp` | 110.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31997824508) |
| #31993851373<br>[#30236 [XPU] Support INT4 dense linear (AWQ/GPTQ) for XPU](https://github.com/sgl-project/sglang/pull/30236) | `int4-linear-xpu` | 106.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31993851373) |
| #29304002213<br>[#31110 [CPU] bypass scoring_func argument in topk for cpu device](https://github.com/sgl-project/sglang/pull/31110) | `fix_topk_interface_change` | 106.3min | 6 | stage-b-test-16-npu-a3, multimodal-gen-test-2-npu-a3, stage-b-test-4-npu-a3, stage-b-test-1-npu-a2 (0), multimodal-gen-test-1-npu-a3, single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29304002213) |
| #29305357370<br>[#30992 support GLM-5.2 MTP index sharing with prefill CP](https://github.com/sgl-project/sglang/pull/30992) | `main` | 103.6min | 5 | stage-b-test-16-npu-a3, stage-b-test-4-npu-a3, multimodal-gen-test-1-npu-a3, multimodal-gen-test-2-npu-a3, single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29305357370) |
| #29305841062<br>[#23534 [XPU] Add XPU device support for LMCache radix cache integration](https://github.com/sgl-project/sglang/pull/23534) | `libinta/xpu_lmcache` | 103.4min | 4 | multimodal-gen-test-2-npu-a3, multimodal-gen-test-1-npu-a3, stage-b-test-4-npu-a3, single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29305841062) |
| #31991991312<br>[#24911 Profiling Enhancements [2/3]: detailed execution step annotations](https://github.com/sgl-project/sglang/pull/24911) | `feat/roofline_annotations` | 101.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31991991312) |
| #31997517074<br>[#31575 Fix rope config compatibility and VL/transformers-fallback weight loading](https://github.com/sgl-project/sglang/pull/31575) | `fix/rope-config-and-vl-weight-loading` | 95.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31997517074) |
| #29305065543<br>[#31038 [XPU] Route topk_sigmoid and topk_softmax to AOT sgl-kernel-xpu symbols](https://github.com/sgl-project/sglang/pull/31038) | `xpu/fix-moe-topk-import` | 86.3min | 5 | multimodal-gen-test-2-npu-a3, stage-b-test-16-npu-a3, stage-b-test-4-npu-a3, multimodal-gen-test-1-npu-a3, single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29305065543) |
| #31999574171<br>[#31370 feat(moe): fold padded-topk_ids fill into fused shared-experts append+remap](https://github.com/sgl-project/sglang/pull/31370) | `feat/fold-pad-fill-into-moe-append-remap` | 78.3min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31999574171) |
| #31999957095<br>[#35062 [Misc] Clean up python/sglang package structure](https://github.com/sgl-project/sglang/pull/35062) | `cleanup-python-sglang-structure` | 76.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31999957095) |
| #32000150473<br>[#35060 Clean up environ.py: remove dead env vars, unify deprecation handling, move examples to a unit test](https://github.com/sgl-project/sglang/pull/35060) | `cleanup-environ` | 71.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32000150473) |
| #31992075138<br>[#35020 [Fix] Correct dense FP8 Marlin bias ordering](https://github.com/sgl-project/sglang/pull/35020) | `main` | 71.4min | 2 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31992075138) |
| #31993060078<br>[#34565 [Unified Tree] Support Branching-Point Caching for the SWA Component](https://github.com/sgl-project/sglang/pull/34565) | `cjc/unified-swa-branching` | 69.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31993060078) |
| #31995620775<br>[#31698 [Scheduler] Reuse per-step cuda events uniformly (WAR read_done + copy_done)](https://github.com/sgl-project/sglang/pull/31698) | `htphan/event-reuse` | 63.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31995620775) |
| #31998401389<br>[#34908 Support Intern-S2-Mobius FP8](https://github.com/sgl-project/sglang/pull/34908) | `docs/intern-s2-mobius-fp8-cookbook` | 61.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31998401389) |
| #31998380426 | `jiayi/fix_bug` | 59.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31998380426) |
| #31996038002<br>[#30519 [AMD] [GLM5] fp8 MLA absorbed bmm for GLM-5.2 on gfx950](https://github.com/sgl-project/sglang/pull/30519) | `jacob/glm-mla-fp8-absorbed-bmm` | 58.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31996038002) |
| #31996700658<br>[#31730 [XPU] Fix Encoder Decoder KV Cache Alignment](https://github.com/sgl-project/sglang/pull/31730) | `dev/skrohit/encoder-decoder-kv-cache-align` | 49.9min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31996700658) |
| #29308457338<br>[#31110 [CPU] bypass scoring_func argument in topk for cpu device](https://github.com/sgl-project/sglang/pull/31110) | `fix_topk_interface_change` | 42.6min | 6 | stage-b-test-4-npu-a3, stage-b-test-16-npu-a3, multimodal-gen-test-2-npu-a3, multimodal-gen-test-1-npu-a3, stage-b-test-1-npu-a2 (0), single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29308457338) |
| #29338045581<br>[#31110 [CPU] bypass scoring_func argument in topk for cpu device](https://github.com/sgl-project/sglang/pull/31110) | `main` | 29.9min | 7 | multimodal-gen-test-2-npu-a3, multimodal-gen-test-1-npu-a3, stage-b-test-16-npu-a3, stage-b-test-4-npu-a3, stage-b-test-1-npu-a2 (0), stage-b-test-1-npu-a2 (1), single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29338045581) |
| #29337178545<br>[#23317 [Bug Fix] Sync FlashInfer autotune tactic selection across TP ranks](https://github.com/sgl-project/sglang/pull/23317) | `htphan/fix-symm-mem-cuda-graph-deadlock` | 28.1min | 7 | stage-b-test-16-npu-a3, multimodal-gen-test-1-npu-a3, stage-b-test-1-npu-a2 (0), stage-b-test-4-npu-a3, multimodal-gen-test-2-npu-a3, stage-b-test-1-npu-a2 (1), single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29337178545) |
| #29309792849<br>[#27375 [Model] Add support for JetBrains' Mellum v2 code generation model](https://github.com/sgl-project/sglang/pull/27375) | `main` | 22.1min | 8 | multimodal-gen-test-1-npu-a3, stage-b-test-16-npu-a3, multimodal-gen-test-2-npu-a3, stage-b-test-1-npu-a2 (0), stage-b-test-4-npu-a3, stage-b-test-1-npu-a2 (1), stage-b-test-2-npu-a2 (1), single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29309792849) |
| #31996737744<br>[#34277 [DSV4] Emit TMA-aligned UE8M0 scales for FP8 einsum](https://github.com/sgl-project/sglang/pull/34277) | `main` | 21.4min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31996737744) |
| #31993946214<br>[#34580 [AMD] Optimize KIMI-K3 with Triton MLA decode kernel by tuning the stage-1 geometry for gfx950](https://github.com/sgl-project/sglang/pull/34580) | `amd-mla-decode-gfx950-tune` | 20.1min | 11 | base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31993946214) |
| #29342967602 | `bbuf/hpc-ops-attention-backend` | 19.9min | 8 | multimodal-gen-test-1-npu-a3, stage-b-test-16-npu-a3, stage-b-test-1-npu-a2 (0), multimodal-gen-test-2-npu-a3, stage-b-test-1-npu-a2 (1), stage-b-test-4-npu-a3, stage-b-test-2-npu-a2 (1), single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29342967602) |
| #31998793918<br>[#34999 [Engine] Freeze GC after server warmup](https://github.com/sgl-project/sglang/pull/34999) | `main` | 13.7min | 12 | base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31998793918) |
| #31993178096 | `amd-mla-decode-gfx950-tune` | 13.6min | 12 | base-b-test-4-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-b-test-4-npu-a3 / run (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31993178096) |
| #29340220767<br>[#30619 [NPU] Fix CPU device for node topology probe](https://github.com/sgl-project/sglang/pull/30619) | `main` | 12.9min | 9 | multimodal-gen-test-1-npu-a3, multimodal-gen-test-2-npu-a3, stage-b-test-16-npu-a3, stage-b-test-4-npu-a3, stage-b-test-2-npu-a2 (1), stage-b-test-2-npu-a2 (0), stage-b-test-1-npu-a2 (0), stage-b-test-1-npu-a2 (1), single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29340220767) |
| #31996077981<br>[#34889 [DCP]Localize HiCache DCP indices once per transfer, not per layer](https://github.com/sgl-project/sglang/pull/34889) | `main` | 12.2min | 12 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31996077981) |
| #31999560740<br>[#35060 Clean up environ.py: remove dead env vars, unify deprecation handling, move examples to a unit test](https://github.com/sgl-project/sglang/pull/35060) | `cleanup-environ` | 10.1min | 12 | base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31999560740) |
| #29338187183 | `bbuf/hpc-ops-attention-backend` | 9.2min | 9 | multimodal-gen-test-2-npu-a3, multimodal-gen-test-1-npu-a3, stage-b-test-2-npu-a2 (0), stage-b-test-16-npu-a3, stage-b-test-1-npu-a2 (0), stage-b-test-1-npu-a2 (1), stage-b-test-4-npu-a3, stage-b-test-2-npu-a2 (1), single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29338187183) |

---


## [Run #29412383315](https://github.com/sgl-project/sglang/actions/runs/29412383315)
- **分支**: `agent/minimax-m3-dead-code`
- **总耗时**: 177.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29412383315

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 58.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995631) |
| stage-b-test-4-npu-a3 | 43.8min | 超时 | 测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995645) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、checkout和upload-artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  - 失败用例: FAILED sglang/multimodal_gen/test/server/ascend/test_server_1_npu.py::TestDiffusionServerOneNpu::test_diffusion_generation[glm_image_t2i_1npu] - Failed: Diffusion testcase 'glm_image_t2i_1npu' failed , FAILED sglang/multimodal_gen/test/server/ascend/test_server_1_npu.py::TestDiffusionServerOneNpu::test_diffusion_generation[glm_image_t2i_1npu] - Failed: Diffusion testcase 'glm_image_t2i_1npu' failed 
  链接: https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995631

- **stage-b-test-4-npu-a3**: test_npu_llada2_mini.py 运行 794 秒后失败，总测试时长 2450 秒，接近超时限制，可能因模型推理或资源竞争导致执行时间过长。
  链接: https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995645

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995688) |
| stage-b-test-1-npu-a2 (1) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995693) |
| multimodal-gen-test-2-npu-a3 | 51.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995708) |
| stage-b-test-1-npu-a2 (0) | 43.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995709) |
| stage-b-test-16-npu-a3 | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995755) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363995882) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29412383315/job/87363996062) |


## [Run #29342967602](https://github.com/sgl-project/sglang/actions/runs/29342967602)
- **分支**: `bbuf/hpc-ops-attention-backend`
- **总耗时**: 19.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29342967602

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431535) |
| stage-b-test-16-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431548) |
| stage-b-test-1-npu-a2 (0) | 18.8min | 环境问题 | NPU测试执行到第二个用例时容器异常退出，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431558) |
| multimodal-gen-test-2-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431583) |
| stage-b-test-1-npu-a2 (1) | 18.5min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431654) |
| stage-b-test-4-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431793) |
| stage-b-test-2-npu-a2 (1) | 18.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431873) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 18.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119432161) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431535

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431548

- **stage-b-test-1-npu-a2 (0)**: 第一个测试用例通过（精度0.870），但第二个用例test_npu_piecewise_graph_prefill.py启动后不久，自定义容器实现执行失败，日志显示“Executing the custom container implementation failed”，属于运行环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431558

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的工件/数据文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431583

- **stage-b-test-1-npu-a2 (1)**: 作业在加载torch_npu和初始化分布式环境后，执行自定义容器实现时失败，报错'Executing the custom container implementation failed'，可能是NPU驱动或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431654

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431793

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行中突然出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431873

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志文件在 Azure Blob 中不存在，可能是日志上传失败或路径错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119432161

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29342967602/job/87119431542) |


## [Run #29340220767](https://github.com/sgl-project/sglang/actions/runs/29340220767)
- **分支**: `main`
- **总耗时**: 12.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29340220767

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868545) |
| multimodal-gen-test-2-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868550) |
| stage-b-test-16-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868574) |
| stage-b-test-4-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868576) |
| stage-b-test-2-npu-a2 (1) | 10.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868586) |
| stage-b-test-2-npu-a2 (0) | 10.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868591) |
| stage-b-test-1-npu-a2 (0) | 11.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868657) |
| stage-b-test-1-npu-a2 (1) | 11.7min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868767) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109869193) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868545

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868550

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868574

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868576

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常（进度条49%），但突然报错"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868586

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行到86%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868591

- **stage-b-test-1-npu-a2 (0)**: 日志显示在TokenizerManager初始化后，出现“Executing the custom container implementation failed”错误，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868657

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试进行到54%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109868767

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/数据文件在存储中缺失，可能是由于文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340220767/job/87109869193


## [Run #29338187183](https://github.com/sgl-project/sglang/actions/runs/29338187183)
- **分支**: `bbuf/hpc-ops-attention-backend`
- **总耗时**: 9.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29338187183

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813163) |
| multimodal-gen-test-1-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813214) |
| stage-b-test-2-npu-a2 (0) | 8.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813231) |
| stage-b-test-16-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813239) |
| stage-b-test-1-npu-a2 (0) | 7.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813270) |
| stage-b-test-1-npu-a2 (1) | 8.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813271) |
| stage-b-test-4-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813277) |
| stage-b-test-2-npu-a2 (1) | 8.3min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813602) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102814028) |

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，请求的 blob 在存储中不存在，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813163

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813214

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行正常（HTTP 200），但随后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813231

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813239

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行中（Prefill batch正常），但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813270

- **stage-b-test-1-npu-a2 (1)**: 作业在运行test_npu_graph_tp1_bf16.py时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813271

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813277

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常（Prefill batch正常处理），但突然报错"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102813602

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338187183/job/87102814028


## [Run #29338045581](https://github.com/sgl-project/sglang/actions/runs/29338045581)
- **分支**: `main`
- **总耗时**: 29.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29338045581

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 28.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417624) |
| multimodal-gen-test-1-npu-a3 | 28.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417631) |
| stage-b-test-16-npu-a3 | 28.8min | 环境问题 | CI 作业因 Azure Blob 存储中指定的 blob 不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417632) |
| stage-b-test-4-npu-a3 | 28.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417656) |
| stage-b-test-1-npu-a2 (0) | 28.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417717) |
| stage-b-test-1-npu-a2 (1) | 28.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417731) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 28.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102418193) |

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417624

- **multimodal-gen-test-1-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417631

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，表明作业尝试下载或访问的 blob 资源已被删除或路径错误，属于环境或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417632

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 资源缺失或路径错误，可能是文件未上传、被删除或配置的 URL 有误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417656

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重后，执行自定义容器实现时失败，报错提示联系自托管runner管理员，属于runner环境问题而非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417717

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行到75%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417731

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是构建产物未上传或存储配置问题，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102418193

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417712) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338045581/job/87102417742) |


## [Run #29337178545](https://github.com/sgl-project/sglang/actions/runs/29337178545)
- **分支**: `htphan/fix-symm-mem-cuda-graph-deadlock`
- **总耗时**: 28.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29337178545

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 27.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352422) |
| multimodal-gen-test-1-npu-a3 | 27.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352459) |
| stage-b-test-1-npu-a2 (0) | 27.2min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352492) |
| stage-b-test-4-npu-a3 | 27.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352508) |
| multimodal-gen-test-2-npu-a3 | 27.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352509) |
| stage-b-test-1-npu-a2 (1) | 27.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352513) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 27.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099353069) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352422

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352459

- **stage-b-test-1-npu-a2 (0)**: 日志显示模型权重加载完成后，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352492

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352508

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352509

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352513

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业日志仅包含一个BlobNotFound错误，表明CI在下载或访问某个依赖文件（如模型权重、测试数据或缓存）时，对应的Azure Blob存储对象缺失或路径错误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099353069

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352441) |
| stage-b-test-2-npu-a2 (1) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337178545/job/87099352526) |


## [Run #29333278063](https://github.com/sgl-project/sglang/actions/runs/29333278063)
- **分支**: `mamba_hicache_fix`
- **总耗时**: 190.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29333278063

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 52.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279795) |
| stage-b-test-4-npu-a3 | 48.2min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279823) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅有checkout、upload-artifact等步骤，且upload-artifact提示无文件上传。可能测试未运行或日志被截断，需查看完整日志。
  - 失败用例: FAILED sglang/multimodal_gen/test/server/ascend/test_server_1_npu.py::TestDiffusionServerOneNpu::test_diffusion_generation[glm_image_t2i_1npu] - Failed: Diffusion testcase 'glm_image_t2i_1npu' failed , FAILED sglang/multimodal_gen/test/server/ascend/test_server_1_npu.py::TestDiffusionServerOneNpu::test_diffusion_generation[glm_image_t2i_1npu] - Failed: Diffusion testcase 'glm_image_t2i_1npu' failed 
  链接: https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279795

- **stage-b-test-4-npu-a3**: 该测试用例在Ascend NPU上执行失败，其余4个测试通过。可能是代码逻辑错误或环境依赖问题，需查看具体错误日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279823

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279762) |
| stage-b-test-16-npu-a3 | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279763) |
| multimodal-gen-test-2-npu-a3 | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279792) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279815) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279869) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086279962) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29333278063/job/87086280516) |


## [Run #29319025946](https://github.com/sgl-project/sglang/actions/runs/29319025946)
- **分支**: `fix_topk_interface_change`
- **总耗时**: 136.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29319025946

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.8min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669286) |
| multimodal-gen-test-1-npu-a3 | 60.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669298) |
| stage-b-test-2-npu-a2 (0) | 6.0min | 环境问题 | pip下载依赖时网络连接中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669309) |
| stage-b-test-1-npu-a2 (0) | 5.7min | 环境问题 | pip下载依赖时网络连接中断导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669341) |
| stage-b-test-2-npu-a2 (1) | 6.0min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669379) |

- **stage-b-test-4-npu-a3**: 作业中5个NPU测试用例，4个通过，仅test_npu_llada2_mini.py失败（exit code 1），耗时869秒，属于该测试用例自身代码或环境问题导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669286

- **multimodal-gen-test-1-npu-a3**: 日志截断于上传工件阶段，未包含测试执行和失败断言信息，无法判断具体失败原因。可能为测试未运行或日志缺失。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669298

- **stage-b-test-2-npu-a2 (0)**: 日志显示pip在下载过程中出现IncompleteRead错误，连接中断，读取字节数不完整，导致依赖安装失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669309

- **stage-b-test-1-npu-a2 (0)**: 在安装Python依赖过程中，pip从网络下载包时出现IncompleteRead错误（已读78MB，还差109MB），连接被中断，导致安装步骤以非零退出码失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669341

- **stage-b-test-2-npu-a2 (1)**: 日志显示模型权重加载完成后，在获取ASCEND_OPP_PATH环境变量时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669379

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 35.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669257) |
| stage-b-test-1-npu-a2 (1) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669300) |
| stage-b-test-16-npu-a3 | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669339) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319025946/job/87039669964) |


## [Run #29318370591](https://github.com/sgl-project/sglang/actions/runs/29318370591)
- **分支**: `ds_v4_xpu_silu_and_mul_clamp`
- **总耗时**: 115.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29318370591

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 4.2min | 环境问题 | pip下载依赖时网络连接中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585156) |
| stage-b-test-2-npu-a2 (0) | 4.3min | 环境问题 | pip安装依赖时网络连接中断导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585192) |
| stage-b-test-4-npu-a3 | 46.7min | 代码错误 | NPU测试中test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585213) |
| stage-b-test-2-npu-a2 (1) | 4.8min | 环境问题 | 自定义容器执行失败，测试未开始即中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585274) |

- **stage-b-test-1-npu-a2 (1)**: 在安装Python依赖包时，pip从网络下载文件过程中连接中断（IncompleteRead），仅读取67MB但预期需要188MB，最终导致命令退出码非零，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585156

- **stage-b-test-2-npu-a2 (0)**: 日志显示pip在下载包时遇到Connection broken: IncompleteRead错误，下载不完整（已读88MB，还需100MB），属于网络不稳定或源问题，非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585192

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，测试文件test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py运行失败（exit code 1），其余4个测试均通过。该测试耗时857秒，可能涉及LLADA2模型在NPU上的功能或兼容性问题，需检查该测试的具体报错日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585213

- **stage-b-test-2-npu-a2 (1)**: 作业在启动第一个测试时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585274

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585188) |
| stage-b-test-1-npu-a2 (0) | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585190) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318370591/job/87037585580) |


## [Run #29317020495](https://github.com/sgl-project/sglang/actions/runs/29317020495)
- **分支**: `repo-cache-dtype->fp32`
- **总耗时**: 127.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29317020495

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 4.5min | 环境问题 | pip下载依赖时网络连接中断导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321539) |
| stage-b-test-4-npu-a3 | 48.2min | 代码错误 | NPU测试中test_npu_llada2_mini.py执行失败，退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321545) |
| stage-b-test-2-npu-a2 (1) | 4.7min | 环境问题 | 自定义容器执行失败，测试在启动后立即中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321585) |
| stage-b-test-2-npu-a2 (0) | 4.5min | 环境问题 | pip安装依赖时网络连接中断导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321609) |
| multimodal-gen-test-1-npu-a3 | 58.5min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321750) |

- **stage-b-test-1-npu-a2 (0)**: 日志显示pip在下载包时出现ProtocolError: Connection broken: IncompleteRead，网络传输中断导致依赖安装失败，属于环境网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321539

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，测试test_npu_llada2_mini.py失败（exit code 1），其余4个测试均通过。该测试属于dllm功能模块，可能因代码逻辑错误或环境依赖问题导致失败，需进一步查看具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321545

- **stage-b-test-2-npu-a2 (1)**: 作业在运行测试test_npu_mla_fia_w8a8int8.py时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321585

- **stage-b-test-2-npu-a2 (0)**: 在安装Python依赖包时，pip下载过程中出现连接中断（IncompleteRead），已读取约105MB但仍有83MB未下载完成，属于网络不稳定或镜像源问题，非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321609

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、checkout和upload-artifact步骤，未显示multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321750

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 45.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321509) |
| stage-b-test-16-npu-a3 | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321521) |
| stage-b-test-1-npu-a2 (1) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321554) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29317020495/job/87037321987) |


## [Run #29311302659](https://github.com/sgl-project/sglang/actions/runs/29311302659)
- **分支**: `topk-glm`
- **总耗时**: 168.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29311302659

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 37.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208265) |
| multimodal-gen-test-1-npu-a3 | 62.5min | 其他 | 日志不完整，未显示测试失败的具体原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208295) |
| stage-b-test-4-npu-a3 | 25.6min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208302) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误或失败信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  - 失败用例: FAILED sglang/multimodal_gen/test/server/ascend/test_server_2_npu.py::TestDiffusionServerTwoNpu::test_diffusion_generation[wan2_2_t2v_14b_w8a8_2npu] - Failed: Diffusion testcase 'wan2_2_t2v_14b_w8a8_2
  链接: https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208265

- **multimodal-gen-test-1-npu-a3**: 作业日志被截断，中间测试执行部分缺失。末尾显示上传diffusion-failures/时无文件，说明测试可能未产生失败样本或提前退出，但无法从现有日志判断具体失败原因。
  - 失败用例: FAILED sglang/multimodal_gen/test/server/ascend/test_server_1_npu.py::TestDiffusionServerOneNpu::test_diffusion_generation[glm_image_t2i_1npu] - Failed: Diffusion testcase 'glm_image_t2i_1npu' failed , FAILED sglang/multimodal_gen/test/server/ascend/test_server_1_npu.py::TestDiffusionServerOneNpu::test_diffusion_generation[glm_image_t2i_1npu] - Failed: Diffusion testcase 'glm_image_t2i_1npu' failed 
  链接: https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208295

- **stage-b-test-4-npu-a3**: 测试文件test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py运行失败，耗时853秒超过预估400秒，最终返回退出码1，导致作业整体失败。具体失败原因需查看该测试的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208302

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208260) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208263) |
| stage-b-test-1-npu-a2 (1) | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208294) |
| stage-b-test-1-npu-a2 (0) | 43.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208315) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208318) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29311302659/job/87070208770) |


## [Run #29309792849](https://github.com/sgl-project/sglang/actions/runs/29309792849)
- **分支**: `main`
- **总耗时**: 22.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29309792849

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 21.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047891) |
| stage-b-test-16-npu-a3 | 21.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047903) |
| multimodal-gen-test-2-npu-a3 | 21.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047906) |
| stage-b-test-1-npu-a2 (0) | 20.8min | 环境问题 | NPU容器在CUDA graph捕获阶段崩溃，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047926) |
| stage-b-test-4-npu-a3 | 21.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047932) |
| stage-b-test-1-npu-a2 (1) | 20.8min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047934) |
| stage-b-test-2-npu-a2 (1) | 20.7min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047955) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 21.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011048145) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047891

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047903

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047906

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型和分配KV cache后，进入CUDA graph捕获阶段时容器异常退出，报错'Executing the custom container implementation failed'，属于NPU环境或运行时稳定性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047926

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047932

- **stage-b-test-1-npu-a2 (1)**: 作业在加载模型权重完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047934

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行到99%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047955

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源已被删除或路径错误，可能是构建产物或测试数据未正确上传，需检查存储配置或重新上传相关文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011048145

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29309792849/job/87011047924) |


## [Run #29308457338](https://github.com/sgl-project/sglang/actions/runs/29308457338)
- **分支**: `fix_topk_interface_change`
- **总耗时**: 42.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29308457338

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 41.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964727) |
| stage-b-test-16-npu-a3 | 41.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964730) |
| multimodal-gen-test-2-npu-a3 | 41.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964736) |
| multimodal-gen-test-1-npu-a3 | 41.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964737) |
| stage-b-test-1-npu-a2 (0) | 41.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964747) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 41.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006965024) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964727

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964730

- **multimodal-gen-test-2-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964736

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964737

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到83%时，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964747

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的模型/数据文件在 Azure Blob 存储中缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006965024

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964734) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964745) |
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29308457338/job/87006964748) |


## [Run #29306153312](https://github.com/sgl-project/sglang/actions/runs/29306153312)
- **分支**: `fix/amd-ci-perf-bounds-and-dispatcher-test`
- **总耗时**: 119.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29306153312

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 29.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836339) |
| stage-b-test-4-npu-a3 | 31.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836373) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因需查看完整日志才能确定。
  链接: https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836339

- **stage-b-test-4-npu-a3**: 日志显示测试运行中（Prefill batch正常），但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836373

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836329) |
| stage-b-test-1-npu-a2 (1) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836335) |
| multimodal-gen-test-2-npu-a3 | 33.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836338) |
| stage-b-test-2-npu-a2 (1) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836342) |
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836361) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836398) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29306153312/job/86999836499) |


## [Run #29305841062](https://github.com/sgl-project/sglang/actions/runs/29305841062)
- **分支**: `libinta/xpu_lmcache`
- **总耗时**: 103.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29305841062

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 17.9min | 其他 | 作业日志被截断，未显示实际失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902627) |
| multimodal-gen-test-1-npu-a3 | 5.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902631) |
| stage-b-test-4-npu-a3 | 9.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902671) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.7min | 其他 | 日志被截断，未显示测试执行结果，仅看到作业清理和Node.js弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902973) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，但未出现明确错误信息，可能为测试失败但未生成失败文件，或日志截断导致关键错误缺失。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902627

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、checkout和upload-artifact步骤，未展示multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902631

- **stage-b-test-4-npu-a3**: 日志显示测试进行中（约51%进度）时，GitHub Actions 报错“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于 runner 或容器环境异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902671

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分被省略，无法看到测试命令的实际输出和失败原因。仅显示作业结束时的清理步骤和Node.js 20弃用警告，无明确错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902973

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 40.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902644) |
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902646) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902664) |
| stage-b-test-1-npu-a2 (1) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902668) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305841062/job/86998902691) |


## [Run #29305357370](https://github.com/sgl-project/sglang/actions/runs/29305357370)
- **分支**: `main`
- **总耗时**: 103.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29305357370

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 14.1min | 环境问题 | NPU服务健康检查返回503，导致自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565825) |
| stage-b-test-4-npu-a3 | 102.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565848) |
| multimodal-gen-test-1-npu-a3 | 102.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565867) |
| multimodal-gen-test-2-npu-a3 | 17.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565881) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 102.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997566084) |

- **stage-b-test-16-npu-a3**: 日志显示服务启动后/health_generate接口返回503 Service Unavailable，说明NPU推理服务未能正常就绪，随后自定义容器执行失败，属于环境或服务启动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565825

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565848

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565867

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565881

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997566084

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565840) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565842) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565854) |
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305357370/job/86997565894) |


## [Run #29305065543](https://github.com/sgl-project/sglang/actions/runs/29305065543)
- **分支**: `xpu/fix-moe-topk-import`
- **总耗时**: 86.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29305065543

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 85.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726417) |
| stage-b-test-16-npu-a3 | 85.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726420) |
| stage-b-test-4-npu-a3 | 85.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726422) |
| multimodal-gen-test-1-npu-a3 | 85.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726434) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 85.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726723) |

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726417

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是构建产物未上传或存储配置变更，需检查相关存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726420

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726422

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726434

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726723

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726445) |
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726448) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726451) |
| stage-b-test-1-npu-a2 (1) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29305065543/job/86996726467) |


## [Run #29304163296](https://github.com/sgl-project/sglang/actions/runs/29304163296)
- **分支**: `remove-qserve-quantization`
- **总耗时**: 165.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29304163296

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.6min | 其他 | 日志被截断，无法确定具体失败原因，仅显示上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055459) |
| stage-b-test-4-npu-a3 | 39.6min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055471) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，仅看到上传diffusion-failures目录时提示无文件，未显示实际测试失败信息，需查看完整日志才能定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055459

- **stage-b-test-4-npu-a3**: 作业中4个NPU测试有3个通过，但test_npu_llada2_mini.py失败（exit code 1），导致整体作业以255退出。该测试属于dllm功能模块，可能涉及代码逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055471

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055460) |
| stage-b-test-2-npu-a2 (0) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055476) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055495) |
| stage-b-test-16-npu-a3 | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055501) |
| multimodal-gen-test-2-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055526) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055579) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304163296/job/86994055857) |


## [Run #29304002213](https://github.com/sgl-project/sglang/actions/runs/29304002213)
- **分支**: `fix_topk_interface_change`
- **总耗时**: 106.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29304002213

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 105.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580860) |
| multimodal-gen-test-2-npu-a3 | 105.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580865) |
| stage-b-test-4-npu-a3 | 105.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580875) |
| stage-b-test-1-npu-a2 (0) | 33.4min | 代码错误 | NPU测试中test_npu_autoround_moe.py失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580885) |
| multimodal-gen-test-1-npu-a3 | 105.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580891) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 105.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993581145) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580860

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580865

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查依赖的构建产物是否正常生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580875

- **stage-b-test-1-npu-a2 (0)**: 测试test/registered/ascend/basic_function/quant/test_npu_autoround_moe.py执行失败（exit code 1），其余3个测试均通过。该测试涉及量化功能，可能是代码逻辑或环境依赖问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580885

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580891

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象缺失，可能是构建产物或依赖文件未正确上传，或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993581145

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580872) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580917) |
| stage-b-test-1-npu-a2 (1) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29304002213/job/86993580933) |


## [Run #32000150473](https://github.com/sgl-project/sglang/actions/runs/32000150473)
- **分支**: `cleanup-environ`
- **总耗时**: 71.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32000150473

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 46.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918639) |
| base-b-test-8-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918722) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918749) |
| base-b-test-4-npu-a3 / run (1) | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918768) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918777) |
| base-b-test-4-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918803) |
| base-b-test-16-npu-a3 / run (0) | 3.0min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918820) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298919044) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.2min | 其他 | 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298919053) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298919055) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298919057) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤，无法判断失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918639

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被取消，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918722

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918749

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行健康检查时，检测到multimodal-gen-test-1-npu-a3作业失败，属于根因失败，因此本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918768

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918777

- **base-b-test-4-npu-a3 / run (0)**: 日志显示本作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被取消，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918803

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918820

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298919044

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示本作业未实际运行测试，而是因PR健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发了fast-fail机制，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298919053

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298919055

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298919057

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32000150473/job/95298918742) |


## [Run #31999957095](https://github.com/sgl-project/sglang/actions/runs/31999957095)
- **分支**: `cleanup-python-sglang-structure`
- **总耗时**: 76.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31999957095

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 38.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350620) |
| base-b-test-4-npu-a3 / run (1) | 2.8min | 其他 | 健康检查发现根因作业失败，导致级联跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350742) |
| base-b-test-4-npu-a3 / run (0) | 4.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350786) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350850) |
| base-b-test-8-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350938) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.0min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350963) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350970) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350978) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298351022) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions的初始化、Node版本警告和上传artifact步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350620

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业被快速失败机制跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350742

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350786

- **base-b-test-16-npu-a3 / run (0)**: 健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而快速失败，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350850

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动前的健康检查中检测到multimodal-gen-test-1-npu-a3作业失败，作为根因作业触发了fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350938

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350963

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业在启动阶段被健康检查快速失败机制终止，原因是根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350970

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于依赖作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350978

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 该作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298351022

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350803) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350854) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999957095/job/95298350874) |


## [Run #31999768419](https://github.com/sgl-project/sglang/actions/runs/31999768419)
- **分支**: `dsv4_fp8_trtllm_gen`
- **总耗时**: 116.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31999768419

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.6min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826508) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 72.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826685) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95307627894) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95311685392) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95315619117) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826508

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826685

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1108秒，未达到性能预期，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95307627894

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，导致本作业在启动前被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95311685392

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动前的PR健康检查阶段，检测到multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，因此触发fast-fail跳过本作业，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95315619117

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826543) |
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826548) |
| base-b-test-4-npu-a3 / run (0) | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826555) |
| base-b-test-16-npu-a3 / run (0) | 46.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826560) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826563) |
| base-b-test-8-npu-a3 / run (0) | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826595) |
| base-b-test-4-npu-a3 / run (1) | 13.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826599) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826700) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 49.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826708) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999768419/job/95297826804) |


## [Run #31999574171](https://github.com/sgl-project/sglang/actions/runs/31999574171)
- **分支**: `feat/fold-pad-fill-into-moe-append-remap`
- **总耗时**: 78.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31999574171

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304768) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297305086) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 38.1min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297305140) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 4.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95311177039) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败产物，但作业仍标记为失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304768

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在执行过程中出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297305086

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在启动容器时出现错误，提示“Executing the custom container implementation failed”，可能是容器镜像拉取失败或运行环境配置问题，导致测试无法正常执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297305140

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95311177039

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304823) |
| base-b-test-4-npu-a3 / run (0) | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304864) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304874) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304880) |
| base-b-test-4-npu-a3 / run (1) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304931) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304943) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297304976) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297305088) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95297305092) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31999574171/job/95305452943) |


## [Run #31999560740](https://github.com/sgl-project/sglang/actions/runs/31999560740)
- **分支**: `cleanup-environ`
- **总耗时**: 10.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31999560740

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 9.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274711) |
| multimodal-gen-test-1-npu-a3 | 8.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274721) |
| base-a-test-1-npu-a2 / run (0) | 9.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274741) |
| base-b-test-16-npu-a3 / run (0) | 9.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274760) |
| base-b-test-4-npu-a3 / run (1) | 9.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274796) |
| base-b-test-4-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274815) |
| base-b-test-2-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274835) |
| base-b-test-8-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274926) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297275071) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297275094) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297275157) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297275282) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274711

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未出现测试执行或失败断言。可能因日志截断或作业在测试前被取消，需查看完整日志定位真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274721

- **base-a-test-1-npu-a2 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误所致，属于基础设施或环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274741

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载日志文件，但 Blob 存储返回 BlobNotFound 错误，可能是日志文件被清理、路径错误或上传失败，属于基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274760

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274796

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274815

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274835

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297274926

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297275071

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297275094

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297275157

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31999560740/job/95297275282


## [Run #31998793918](https://github.com/sgl-project/sglang/actions/runs/31998793918)
- **分支**: `main`
- **总耗时**: 13.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31998793918

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 12.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295236963) |
| base-b-test-2-npu-a3 / run (0) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237003) |
| multimodal-gen-test-1-npu-a3 | 11.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237011) |
| base-b-test-16-npu-a3 / run (0) | 12.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237024) |
| base-b-test-4-npu-a3 / run (0) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237027) |
| base-b-test-4-npu-a3 / run (1) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237029) |
| base-b-test-1-npu-a3 / run (0) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237045) |
| base-a-test-1-npu-a2 / run (0) | 12.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237068) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237158) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237232) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237260) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237281) |

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、资源被清理或配置变更所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295236963

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237003

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败产物，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237011

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237024

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件已被删除或路径错误，属于基础设施/存储配置问题，需检查相关 blob 是否存在或更新下载链接。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237027

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237029

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237045

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237068

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237158

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237232

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查相关 blob 是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237260

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998793918/job/95295237281


## [Run #31998401389](https://github.com/sgl-project/sglang/actions/runs/31998401389)
- **分支**: `docs/intern-s2-mobius-fp8-cookbook`
- **总耗时**: 61.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31998401389

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 52.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184567) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184599) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184627) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184643) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184655) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184670) |
| base-b-test-1-npu-a3 / run (0) | 1.8min | 其他 | 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184688) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184748) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184766) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.4min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184787) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184837) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、Node版本警告及上传artifact时未找到diffusion-failures目录。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184567

- **base-b-test-8-npu-a3 / run (0)**: 作业启动后健康检查发现multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184599

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184627

- **base-b-test-2-npu-a3 / run (0)**: 本作业在健康检查阶段发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184643

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查识别出根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3）因级联失败被过滤并快速失败，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184655

- **base-b-test-4-npu-a3 / run (1)**: 本作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，主动跳过执行，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184670

- **base-b-test-1-npu-a3 / run (0)**: 本作业本身未执行测试，因健康检查脚本检测到根因作业multimodal-gen-test-1-npu-a3失败，触发了fast-fail机制，跳过所有依赖作业并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184688

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184748

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184766

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184787

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业未实际运行即被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184837

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31998401389/job/95294184716) |


## [Run #31998380426](https://github.com/sgl-project/sglang/actions/runs/31998380426)
- **分支**: `jiayi/fix_bug`
- **总耗时**: 59.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31998380426

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 54.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117122) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117165) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117225) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117233) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117239) |
| base-b-test-16-npu-a3 / run (0) | 2.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117247) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117339) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117545) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.7min | 其他 | 健康检查快速失败，因其他作业（multimodal-gen-test-1-npu-a3）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117694) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117718) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117737) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行约54分钟后结束，上传diffusion-failures目录时提示无文件，说明测试可能通过或失败未产生产物。日志中间部分被省略，无法定位具体错误，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117122

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117165

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117225

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败策略被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117233

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117239

- **base-b-test-16-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117247

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败（fast-fail）被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117339

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因Fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117545

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 本作业在启动前的PR健康检查阶段被快速失败机制终止，原因是根因作业multimodal-gen-test-1-npu-a3已失败，本作业并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117694

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117718

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，判定为根因作业，因此本作业（base-c-test-acc-4-npu-a3）被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117737

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31998380426/job/95294117228) |


## [Run #31997824508](https://github.com/sgl-project/sglang/actions/runs/31997824508)
- **分支**: `tmp/eagle-mtp`
- **总耗时**: 110.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31997824508

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 56.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623420) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 69.1min | 精度回归 | NPU精度测试qwen3_5_9b_bf16_1p_gsm8k失败，0/3通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623744) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 50.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95302052292) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95304697895) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623420

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py运行3903秒后退出码1，所有3个测试均未通过，属于精度回归问题，可能由模型输出与预期不符导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623744

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试指标已正常输出（TPOT 47.82ms，吞吐6075.82），但随后容器执行报错，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95302052292

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95304697895

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623493) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623513) |
| base-b-test-4-npu-a3 / run (0) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623533) |
| base-b-test-2-npu-a3 / run (0) | 19.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623534) |
| base-b-test-16-npu-a3 / run (0) | 50.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623535) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623537) |
| base-b-test-4-npu-a3 / run (1) | 12.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623613) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623710) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623747) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95292623748) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997824508/job/95297886774) |


## [Run #31997517074](https://github.com/sgl-project/sglang/actions/runs/31997517074)
- **分支**: `fix/rope-config-and-vl-weight-loading`
- **总耗时**: 95.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31997517074

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 61.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802598) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 60.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291803143) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 性能回归 | NPU性能测试未达预期，minimax_m2_5测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95297949720) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 34.1min | 性能回归 | NPU性能测试中kimi_k2_6用例失败，未达性能目标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95301232606) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95304492504) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败产物，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802598

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291803143

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1097秒后失败，0/1通过，属于性能指标未达标导致的回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95297949720

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件4个用例中1个失败，kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms.py返回退出码1，耗时753秒，未通过性能测试，其余用例通过，判定为性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95301232606

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业启动前的健康检查检测到同一次PR运行中的multimodal-gen-test-1-npu-a3作业失败，根据快速失败策略，本作业被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95304492504

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802803) |
| base-b-test-16-npu-a3 / run (0) | 52.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802842) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802893) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802913) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802952) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291802969) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291803041) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291803148) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291803203) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997517074/job/95291803291) |


## [Run #31997280258](https://github.com/sgl-project/sglang/actions/runs/31997280258)
- **分支**: `yangminl/agentx-decode-gap-v2-shared-overlap-v2-20260815`
- **总耗时**: 126.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31997280258

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 55.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862842) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.2min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95303403124) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95304689561) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95316650835) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅显示上传diffusion-failures产物时未找到文件，可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862842

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95303403124

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95304689561

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95316650835

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862737) |
| base-b-test-4-npu-a3 / run (1) | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862753) |
| base-b-test-16-npu-a3 / run (0) | 59.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862791) |
| base-b-test-4-npu-a3 / run (0) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862798) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862820) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862834) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293862870) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293863051) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293863087) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293863148) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95293863230) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31997280258/job/95298261420) |


## [Run #31996737744](https://github.com/sgl-project/sglang/actions/runs/31996737744)
- **分支**: `main`
- **总耗时**: 21.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31996737744

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 20.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726644) |
| base-b-test-4-npu-a3 / run (0) | 20.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726662) |
| base-b-test-4-npu-a3 / run (1) | 20.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726719) |
| base-b-test-8-npu-a3 / run (0) | 20.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726727) |
| base-b-test-1-npu-a3 / run (0) | 20.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726795) |
| base-b-test-16-npu-a3 / run (0) | 20.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726800) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289727448) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289727484) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 20.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289727602) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289727633) |

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726644

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726662

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726719

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败、过期或配置错误，需检查相关存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726727

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726795

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726800

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289727448

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289727484

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289727602

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289727633

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31996737744/job/95289726606) |


## [Run #31996700658](https://github.com/sgl-project/sglang/actions/runs/31996700658)
- **分支**: `dev/skrohit/encoder-decoder-kv-cache-align`
- **总耗时**: 49.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31996700658

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 32.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289592960) |
| base-a-test-1-npu-a2 / run (0) | 1.4min | 其他 | 健康检查中的lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593029) |
| base-b-test-4-npu-a3 / run (0) | 0.9min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593044) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593050) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593063) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593084) |
| base-b-test-4-npu-a3 / run (1) | 1.0min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593108) |
| base-b-test-16-npu-a3 / run (0) | 2.4min | 环境问题 | 健康检查中lint检查失败导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593172) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.1min | 其他 | 健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593184) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现lint检查失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593189) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | PR健康检查失败，lint检查未通过导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593193) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | PR健康检查中的lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593233) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行约31分钟后结束，上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本或提前退出。中间日志被省略，无法定位具体错误，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289592960

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593029

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint 检查结论为 failure，触发了 fast-fail 机制，作业未进入实际测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593044

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593050

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业未进入实际测试即终止。这属于CI前置检查失败，而非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593063

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业未进入实际测试即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593084

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593108

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593172

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查状态为failure，触发fast-fail机制，作业未进入实际测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593184

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发了fast-fail机制，作业未开始实际测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593189

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查状态为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593193

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查状态为failure，触发了fast-fail机制，作业未实际运行测试即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996700658/job/95289593233


## [Run #31996077981](https://github.com/sgl-project/sglang/actions/runs/31996077981)
- **分支**: `main`
- **总耗时**: 12.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31996077981

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911013) |
| base-a-test-1-npu-a2 / run (0) | 11.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911093) |
| base-b-test-4-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911115) |
| base-b-test-4-npu-a3 / run (1) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911126) |
| base-b-test-1-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911165) |
| base-b-test-16-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911172) |
| base-b-test-8-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911328) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911387) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911388) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911406) |
| base-b-test-2-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911427) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911446) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未包含multimodal-gen测试的实际执行结果或错误输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911013

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是缓存、模型权重或日志文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911093

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911115

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911126

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除或路径错误，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911165

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911172

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911328

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911387

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911388

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或被误删，需检查存储配置及文件存在性。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911406

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911427

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996077981/job/95287911446


## [Run #31996038002](https://github.com/sgl-project/sglang/actions/runs/31996038002)
- **分支**: `jacob/glm-mla-fp8-absorbed-bmm`
- **总耗时**: 58.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31996038002

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 36.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797516) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797585) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797658) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797732) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查检测到根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797756) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797757) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797771) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.9min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797904) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.0min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联取消 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797972) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797999) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.4min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287798006) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions运行环境准备、Node版本警告及上传artifact时未找到失败文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797516

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797585

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤掉级联失败后，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797658

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发Fast-fail机制，本作业未实际运行即被终止，属于CI依赖链上的连锁失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797732

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被终止，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797756

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797757

- **base-b-test-4-npu-a3 / run (1)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤后仍被快速失败机制终止，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797771

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797904

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业在健康检查阶段被快速失败机制取消，根因是multimodal-gen-test-1-npu-a3作业失败，本作业并非自身问题，而是级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797972

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797999

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤掉级联失败后，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287798006

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31996038002/job/95287797657) |


## [Run #31995620775](https://github.com/sgl-project/sglang/actions/runs/31995620775)
- **分支**: `htphan/event-reuse`
- **总耗时**: 63.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31995620775

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 45.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676414) |
| base-b-test-16-npu-a3 / run (0) | 2.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676448) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676452) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业根因失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676467) |
| base-b-test-2-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676522) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676577) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676583) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676631) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676644) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676679) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 作业因上游根因任务失败被快速失败跳过，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676687) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行过程或失败断言，仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676414

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676448

- **base-b-test-4-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，未执行实际测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676452

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676467

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-2-npu-a3）被快速失败跳过，并非自身执行出错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676522

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676577

- **base-b-test-4-npu-a3 / run (1)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，按策略快速失败，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676583

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676631

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676644

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤掉多个级联失败后，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676679

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因是multimodal-gen-test-1-npu-a3任务失败，导致本作业被Fast-fail跳过，属于上游任务引发的级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676687

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995620775/job/95286676564) |


## [Run #31995334699](https://github.com/sgl-project/sglang/actions/runs/31995334699)
- **分支**: `cheng/gc-s12-carrier`
- **总耗时**: 113.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31995334699

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 53.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914782) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 59.4min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285915078) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 10.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95300735750) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 11.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95301456911) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914782

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在加载模型shards约52%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境或容器问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285915078

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在性能测试运行中途报错"Executing the custom container implementation failed"，提示联系runner管理员，属于NPU自托管runner的容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95300735750

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示测试运行正常，但在执行过程中出现错误："Executing the custom container implementation failed"，提示联系自托管 runner 管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95301456911

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914795) |
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914806) |
| base-b-test-16-npu-a3 / run (0) | 44.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914824) |
| base-b-test-2-npu-a3 / run (0) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914848) |
| base-b-test-4-npu-a3 / run (0) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914861) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914899) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285914909) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 37.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285915012) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285915111) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95285915127) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31995334699/job/95294851333) |


## [Run #31994368156](https://github.com/sgl-project/sglang/actions/runs/31994368156)
- **分支**: `fix/bcg-deepstack-replay-slot`
- **总耗时**: 161.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31994368156

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 59.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370027) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95294199187) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因其他根因作业失败而被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95297288821) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95300274042) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.6min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95308459927) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370027

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，根据快速失败策略，本作业被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95294199187

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，本作业被级联跳过，日志中无自身测试执行或失败信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95297288821

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95300274042

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在“Check PR test health”步骤因multimodal-gen-test-1-npu-a3作业失败而触发fast-fail，属于级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95308459927

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283369994) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370015) |
| base-b-test-4-npu-a3 / run (1) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370045) |
| base-b-test-16-npu-a3 / run (0) | 45.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370056) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370090) |
| base-b-test-4-npu-a3 / run (0) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370116) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370120) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370288) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370320) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370343) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994368156/job/95283370348) |


## [Run #31994353942](https://github.com/sgl-project/sglang/actions/runs/31994353942)
- **分支**: `enable-lora-xpu`
- **总耗时**: 185.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31994353942

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 44.8min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334713) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334752) |
| base-b-test-16-npu-a3 / run (0) | 2.2min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334825) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334846) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334881) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.7min | 其他 | 该作业因其他根因作业失败被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334977) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283335001) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 131.4min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，其他两个测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283335046) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业根因失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95297550242) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言信息，仅显示上传diffusion-failures产物时未找到文件，可能测试未运行或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334713

- **base-b-test-4-npu-a3 / run (1)**: 本作业在启动前的健康检查中发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334752

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行PR健康检查时，检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334825

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身执行出错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334846

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败策略被取消，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334881

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业被标记为级联失败并快速跳过，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334977

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283335001

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 退出码为1，未通过精度验证，而 moonlight 和 glm4 测试均通过，表明该模型存在精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283335046

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95297550242

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334739) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334769) |
| base-b-test-2-npu-a3 / run (0) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283334784) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994353942/job/95283335058) |


## [Run #31994038554](https://github.com/sgl-project/sglang/actions/runs/31994038554)
- **分支**: `codex/diffusion-reuse-srt-clip`
- **总耗时**: 156.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31994038554

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志被截断，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282543871) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95295952674) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95297452142) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95305256477) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行过程或失败原因。仅显示上传artifact时未找到diffusion-failures目录，以及Node 20弃用警告，但无明确错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282543871

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95295952674

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95297452142

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95305256477

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282543983) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282543990) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544017) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544084) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544097) |
| base-b-test-16-npu-a3 / run (0) | 48.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544130) |
| base-b-test-4-npu-a3 / run (1) | 13.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544166) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544402) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544412) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544437) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95282544571) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31994038554/job/95292290362) |


## [Run #31993946214](https://github.com/sgl-project/sglang/actions/runs/31993946214)
- **分支**: `amd-mla-decode-gfx950-tune`
- **总耗时**: 20.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31993946214

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709702) |
| base-b-test-4-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709730) |
| multimodal-gen-test-1-npu-a3 | 10.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709733) |
| base-b-test-4-npu-a3 / run (1) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709759) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709769) |
| base-b-test-16-npu-a3 / run (0) | 19.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709814) |
| base-b-test-1-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709823) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315710098) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315710148) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315710165) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315710235) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709702

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709730

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions的常规准备、上传artifact（无文件）和清理步骤。无法判断是性能、精度还是代码问题，可能因日志截断或作业在测试前已失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709733

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709759

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709769

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储资源缺失，可能是日志上传或依赖下载路径错误，属于基础设施或配置问题，与代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709814

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709823

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315710098

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315710148

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315710165

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315710235

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993946214/job/95315709816) |


## [Run #31993937733](https://github.com/sgl-project/sglang/actions/runs/31993937733)
- **分支**: `clean-dsv4`
- **总耗时**: 144.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31993937733

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.7min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239099) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95289290516) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | PR健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95292023466) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95294034546) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95303138203) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239099

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业启动前的健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业被跳过未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95289290516

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95292023466

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95294034546

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示multimodal-gen-test-1-npu-a3作业失败，被识别为根因失败，本作业作为级联失败被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95303138203

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239101) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239120) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239126) |
| base-b-test-2-npu-a3 / run (0) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239152) |
| base-b-test-16-npu-a3 / run (0) | 58.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239164) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239175) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239304) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 34.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239445) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239535) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239585) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993937733/job/95282239586) |


## [Run #31993851373](https://github.com/sgl-project/sglang/actions/runs/31993851373)
- **分支**: `int4-linear-xpu`
- **总耗时**: 106.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31993851373

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056271) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.8min | 精度回归 | NPU精度测试中qwen3_5_9b用例失败，导致整体测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056566) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95291180377) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95291558645) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败断言信息，仅显示上传diffusion-failures工件时未找到文件，无法判断具体失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056271

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中1/3通过，qwen3_5_9b_bf16_1p_gsm8k用例返回退出码1，耗时3905秒，可能因精度不达标或运行错误导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056566

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95291180377

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段被健康检查拦截，检测到同PR中multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95291558645

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 64.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056358) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056375) |
| base-b-test-2-npu-a3 / run (0) | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056390) |
| base-a-test-1-npu-a2 / run (0) | 12.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056415) |
| base-b-test-4-npu-a3 / run (1) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056435) |
| base-b-test-4-npu-a3 / run (0) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056445) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056527) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056541) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056558) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95282056590) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31993851373/job/95286485779) |


## [Run #31993178096](https://github.com/sgl-project/sglang/actions/runs/31993178096)
- **分支**: `amd-mla-decode-gfx950-tune`
- **总耗时**: 13.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31993178096

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228865) |
| multimodal-gen-test-1-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228867) |
| base-b-test-8-npu-a3 / run (0) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228901) |
| base-a-test-1-npu-a2 / run (0) | 13.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228913) |
| base-b-test-1-npu-a3 / run (0) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228950) |
| base-b-test-2-npu-a3 / run (0) | 13.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228982) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280229126) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280229149) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280229219) |
| base-b-test-16-npu-a3 / run (0) | 13.0min | 环境问题 | CI日志存储的blob不存在，导致无法获取作业日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280229786) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280230318) |
| base-b-test-4-npu-a3 / run (1) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280230448) |

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228865

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或数据在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228867

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 Azure Blob 返回 BlobNotFound 错误，说明 CI 作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228901

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228913

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228950

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280228982

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280229126

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储账户中找到，可能是资源被清理、路径错误或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280229149

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280229219

- **base-b-test-16-npu-a3 / run (0)**: Azure Blob存储返回BlobNotFound错误，说明日志文件已被删除或路径错误，属于基础设施/存储问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280229786

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280230318

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993178096/job/95280230448


## [Run #31993060078](https://github.com/sgl-project/sglang/actions/runs/31993060078)
- **分支**: `cjc/unified-swa-branching`
- **总耗时**: 69.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31993060078

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 44.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279926968) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279926990) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查中lint检查失败导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279926998) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 环境问题 | 健康检查失败：lint检查未通过导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927044) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927065) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927071) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927083) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927197) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.4min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927291) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查失败：lint检查未通过导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927346) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查发现lint检查失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927390) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.0min | 其他 | 健康检查发现lint检查失败，导致作业快速失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927392) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures工件时未找到文件，说明测试可能未产生失败样本或日志被截断，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279926968

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279926990

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279926998

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927044

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927065

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行lint健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927071

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行lint检查时失败（conclusion=failure），触发了fast-fail机制，作业未进入实际测试阶段即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927083

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行lint健康检查时失败（conclusion=failure），触发了fast-fail机制，作业未进入实际测试阶段即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927197

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查结论为failure，触发fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927291

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行健康检查时，检测到PR的lint检查状态为failure，触发了fast-fail机制，作业在运行测试前即被终止，属于前置检查拦截，非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927346

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查状态为failure，触发fast-fail机制，作业未进入实际测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927390

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行PR健康检查时，检测到lint检查状态为failure，触发fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31993060078/job/95279927392


## [Run #31992075138](https://github.com/sgl-project/sglang/actions/runs/31992075138)
- **分支**: `main`
- **总耗时**: 71.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31992075138

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352508) |
| base-b-test-8-npu-a3 / run (0) | 0.9min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352648) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352656) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因作业为multimodal-gen-test-1-npu-a3 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352686) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352692) |
| base-a-test-1-npu-a2 / run (0) | 70.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352791) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查快速失败，根因作业为multimodal-gen-test-1-npu-a3 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352820) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352849) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352953) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352986) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277353005) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277353014) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未包含multimodal-gen测试的具体执行输出或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352508

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352648

- **base-b-test-16-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352656

- **base-b-test-1-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而触发快速失败机制，并非本作业自身问题，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352686

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3）因级联失败被过滤后仍被快速失败机制终止，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352692

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是上传失败或路径配置错误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352791

- **base-b-test-2-npu-a3 / run (0)**: 该作业因其他作业（multimodal-gen-test-1-npu-a3）失败而被级联跳过，属于健康检查快速失败机制，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352820

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352849

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业未实际运行，健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，跳过当前作业并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352953

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被过滤并快速失败，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277352986

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277353005

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31992075138/job/95277353014


## [Run #31991991312](https://github.com/sgl-project/sglang/actions/runs/31991991312)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 101.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31991991312

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 45.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277121925) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277121968) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122003) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122019) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122032) |
| base-b-test-16-npu-a3 / run (0) | 2.7min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122053) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122060) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 环境问题 | 健康检查发现根因作业失败，触发级联跳过导致本作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122120) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.7min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122252) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败（fast-fail）跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122261) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122314) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，根因作业为multimodal-gen-test-1-npu-a3，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95288827046) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、失败断言或错误堆栈，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277121925

- **base-b-test-1-npu-a3 / run (0)**: 本作业在启动前的健康检查中检测到根因失败作业multimodal-gen-test-1-npu-a3，因此被快速失败机制跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277121968

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122003

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122019

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122032

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122053

- **base-b-test-2-npu-a3 / run (0)**: 本作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而提前终止，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122060

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122120

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122252

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122261

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业在启动前的PR健康检查中检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业被跳过未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122314

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95288827046

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 34.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31991991312/job/95277122266) |


## [Run #31991908670](https://github.com/sgl-project/sglang/actions/runs/31991908670)
- **分支**: `feat/kv-events-component-placement-v2`
- **总耗时**: 132.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31991908670

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 36.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276905964) |
| base-b-test-16-npu-a3 / run (0) | 13.4min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906056) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906096) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906292) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906295) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95285139795) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95296719690) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或已通过，但作业被标记为失败，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276905964

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被Fast-fail跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906056

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被Fast-fail跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906096

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906292

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906295

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段因PR健康检查失败被跳过，根因是multimodal-gen-test-1-npu-a3作业失败，本作业属于级联失败，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95285139795

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业在启动阶段因PR健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95296719690

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906073) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906078) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906088) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906093) |
| base-b-test-4-npu-a3 / run (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906117) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906202) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31991908670/job/95276906208) |


---
*Auto-generated by npu_pr_monitor.py*