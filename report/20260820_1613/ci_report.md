# NPU CI 执行监控
**生成时间**: 2026-08-20 08:13 UTC
**分析 Run 数**: 22

---

## 📊 本次执行总结

- **成功 Job 数**: 11
- **失败 Run 数**: 22
- **成功 Job 平均耗时**: 6.1min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-a-test-1-npu-a2 / run (0) | 9.4min | #32325597578 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204486) |
| base-a-test-1-npu-a2 / run (0) | 7.8min | #32327534719 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859290) |
| base-a-test-1-npu-a2 / run (0) | 6.8min | #32327484201 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627486) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | #32324696338 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552684) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | #32332519141 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822630) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | #32328794292 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418110) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | #32328523714 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642378) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | #32325014239 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466305) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | #32324979757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357313) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | #32332934409 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010993) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 18 | 0 | 7 | 0 | 11 |
| base-a-test-1-npu-a2 / run (0) | 14 | 11 | 0 | 3 | 0 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32324696338<br>[#28403 [PD] Introduce runtime role switching between prefill and decode](https://github.com/sgl-project/sglang/pull/28403) | `feat/pd-role-switch-mori-v2` | 288.9min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32324696338) |
| #32325014239<br>[#35457 [AMD][Spec] Pack AITER target-verify GQA for Qwen3.5](https://github.com/sgl-project/sglang/pull/35457) | `feat/aiter-verify-gqa-pack` | 200.6min | 11 | base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32325014239) |
| #32324979757<br>[#35593 [Fix] Support 128-aligned hidden sizes in the W4AFP8 DeepEP low-latency requant kernel](https://github.com/sgl-project/sglang/pull/35593) | `fix/w4a8-ll-per-token-quant-k-tail` | 166.6min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32324979757) |
| #32324615439<br>[#35511 [diffusion] CI: add minimax-h3 ref2va audio consistency coverage and guard peak vram](https://github.com/sgl-project/sglang/pull/35511) | `codex/minimax-h3-ref2va-audio-ci` | 112.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32324615439) |
| #32328300726<br>[#35613 [diffusion] refactor: scope model-specific API parameters](https://github.com/sgl-project/sglang/pull/35613) | `codex/longcat-docs-api` | 107.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32328300726) |
| #32328811065<br>[#35626 [diffusion] fix: keep large vocab tables in host memory under layerwise offload](https://github.com/sgl-project/sglang/pull/35626) | `fix/host-resident-vocab-embedding` | 106.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32328811065) |
| #32328300584<br>[#35618 [diffusion] UX: report where a component's weights are](https://github.com/sgl-project/sglang/pull/35618) | `fix/component-residency-report` | 98.1min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32328300584) |
| #32333285979<br>[#35267 [XPU] Add diffusion server test config for Intel XPU B60](https://github.com/sgl-project/sglang/pull/35267) | `xpu-mmgen-test-config` | 84.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32333285979) |
| #32327534719<br>[#35197 fix(kernel) Fix Helion small-token prefill bug](https://github.com/sgl-project/sglang/pull/35197) | `main` | 62.0min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-b-test-4-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32327534719) |
| #32328794292<br>[#35592 [Fix] Decide SWA slot liveness on the host instead of a device-side mask](https://github.com/sgl-project/sglang/pull/35592) | `lsyin/swa-free-alive` | 61.4min | 11 | base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32328794292) |
| #32325597578<br>[#35592 [Fix] Decide SWA slot liveness on the host instead of a device-side mask](https://github.com/sgl-project/sglang/pull/35592) | `lsyin/swa-free-alive` | 53.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32325597578) |
| #32326018417 | `dev` | 37.0min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32326018417) |
| #32331066667<br>[#35612 [diffusion] fix: keep Cosmos3 T=1 fusion on Blackwell](https://github.com/sgl-project/sglang/pull/35612) | `main` | 31.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32331066667) |
| #32332934409<br>[#34546 [XPU] Fix/kimi linear xpu](https://github.com/sgl-project/sglang/pull/34546) | `main` | 21.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32332934409) |
| #32324727239<br>[#35337 [XPU][CI] key persistent JIT kernel cache by image content ID](https://github.com/sgl-project/sglang/pull/35337) | `main` | 17.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32324727239) |
| #32327484201<br>[#34337 [Spec][LoRA] Support multi-adapter LoRA with EAGLE/NEXTN/DFLASH/DSPARK speculative decoding](https://github.com/sgl-project/sglang/pull/34337) | `lora-spec-multi-adapter` | 17.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32327484201) |
| #32327311611<br>[#35613 [diffusion] refactor: scope model-specific API parameters](https://github.com/sgl-project/sglang/pull/35613) | `codex/longcat-docs-api` | 16.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32327311611) |
| #32327427544<br>[#35618 [diffusion] UX: report where a component's weights are](https://github.com/sgl-project/sglang/pull/35618) | `fix/component-residency-report` | 14.9min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32327427544) |
| #32332519141<br>[#35592 [Fix] Decide SWA slot liveness on the host instead of a device-side mask](https://github.com/sgl-project/sglang/pull/35592) | `lsyin/swa-free-alive` | 14.2min | 10 | base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32332519141) |
| #32328523714<br>[#34337 [Spec][LoRA] Support multi-adapter LoRA with EAGLE/NEXTN/DFLASH/DSPARK speculative decoding](https://github.com/sgl-project/sglang/pull/34337) | `lora-spec-multi-adapter` | 9.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32328523714) |
| #32329067739<br>[#34337 [Spec][LoRA] Support multi-adapter LoRA with EAGLE/NEXTN/DFLASH/DSPARK speculative decoding](https://github.com/sgl-project/sglang/pull/34337) | `lora-spec-multi-adapter` | 7.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32329067739) |
| #32326719815<br>[#28639 [Perf][DSV4] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](https://github.com/sgl-project/sglang/pull/28639) | `dsv4_overlap` | 5.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32326719815) |

---


## [Run #32333285979](https://github.com/sgl-project/sglang/actions/runs/32333285979)
- **分支**: `xpu-mmgen-test-config`
- **总耗时**: 84.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32333285979

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32333285979/job/96317962431) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或未产生失败文件，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32333285979/job/96317962431


## [Run #32332934409](https://github.com/sgl-project/sglang/actions/runs/32332934409)
- **分支**: `main`
- **总耗时**: 21.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32332934409

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010815) |
| base-b-test-4-npu-a3 / run (1) | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010840) |
| base-b-test-1-npu-a3 / run (0) | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010847) |
| base-b-test-8-npu-a3 / run (0) | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010916) |
| base-b-test-16-npu-a3 / run (0) | 20.9min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010929) |
| base-b-test-4-npu-a3 / run (0) | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010954) |
| base-b-test-2-npu-a3 / run (0) | 20.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011098) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011139) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011176) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011178) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011399) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010815

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查相关 blob 是否存在及路径配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010840

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010847

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010916

- **base-b-test-16-npu-a3 / run (0)**: 作业日志显示BlobNotFound错误，表明CI流程尝试下载的某个blob（如模型权重、测试数据或缓存）已被删除或路径错误，属于环境配置或资源缺失问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010929

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或缓存未正确上传，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010954

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011098

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011139

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011176

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011178

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重、数据或缓存）已被删除或路径错误，需检查相关资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317011399

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32332934409/job/96317010993) |


## [Run #32332519141](https://github.com/sgl-project/sglang/actions/runs/32332519141)
- **分支**: `lsyin/swa-free-alive`
- **总耗时**: 14.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32332519141

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 13.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822509) |
| base-b-test-4-npu-a3 / run (1) | 13.4min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822518) |
| base-b-test-16-npu-a3 / run (0) | 13.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822526) |
| base-b-test-1-npu-a3 / run (0) | 13.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822532) |
| base-b-test-2-npu-a3 / run (0) | 13.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822539) |
| base-b-test-8-npu-a3 / run (0) | 13.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822570) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822714) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 13.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822734) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822817) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315823067) |

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822509

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是上游产物未上传或过期清理所致，需检查相关依赖文件是否仍存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822518

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822526

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822532

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822539

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822570

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822714

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822734

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822817

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315823067

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32332519141/job/96315822630) |


## [Run #32331066667](https://github.com/sgl-project/sglang/actions/runs/32331066667)
- **分支**: `main`
- **总耗时**: 31.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32331066667

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32331066667/job/96311692239) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32331066667/job/96311692239


## [Run #32329067739](https://github.com/sgl-project/sglang/actions/runs/32329067739)
- **分支**: `lora-spec-multi-adapter`
- **总耗时**: 7.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32329067739

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161388) |
| base-b-test-8-npu-a3 / run (0) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161392) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | 其他 | 作业实际成功，日志显示所有测试通过，无失败迹象。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161440) |
| base-b-test-4-npu-a3 / run (1) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161443) |
| base-b-test-4-npu-a3 / run (0) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161454) |
| base-b-test-16-npu-a3 / run (0) | 6.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161489) |
| base-b-test-2-npu-a3 / run (0) | 6.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161492) |
| base-b-test-1-npu-a3 / run (0) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161539) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161687) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161694) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161742) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161776) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161388

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161392

- **base-a-test-1-npu-a2 / run (0)**: 日志显示42个测试全部通过，2个测试文件均PASSED，作业正常结束。可能为误报或外部原因导致作业标记为失败，但日志中无任何错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161440

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161443

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关依赖资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161454

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161489

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161492

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161539

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是上传失败、清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161687

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重、缓存或构建产物）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161694

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161742

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储账户中找到，可能是资源被清理、路径错误或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32329067739/job/96306161776


## [Run #32328811065](https://github.com/sgl-project/sglang/actions/runs/32328811065)
- **分支**: `fix/host-resident-vocab-embedding`
- **总耗时**: 106.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32328811065

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328811065/job/96305491176) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体错误或失败断言，仅有Node 20弃用警告和diffusion-failures目录无文件上传的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328811065/job/96305491176


## [Run #32328794292](https://github.com/sgl-project/sglang/actions/runs/32328794292)
- **分支**: `lsyin/swa-free-alive`
- **总耗时**: 61.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32328794292

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418047) |
| base-b-test-4-npu-a3 / run (0) | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418079) |
| base-b-test-4-npu-a3 / run (1) | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418104) |
| base-b-test-1-npu-a3 / run (0) | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418115) |
| multimodal-gen-test-1-npu-a3 | 4.0min | 环境问题 | Git 拉取代码失败，远程仓库缺少指定 commit。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418172) |
| base-b-test-8-npu-a3 / run (0) | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418190) |
| base-b-test-16-npu-a3 / run (0) | 60.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418193) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418330) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418366) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418444) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 60.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418483) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418047

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418079

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418104

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418115

- **multimodal-gen-test-1-npu-a3**: 作业在 checkout 阶段执行 git fetch 时，远程仓库返回 'not our ref'，多次重试均失败，导致无法获取代码。可能是 PR 分支被删除或 commit 不存在，属于基础设施/仓库状态问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418172

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418190

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI依赖的某个存储对象缺失，可能是上传失败、路径错误或资源被清理，属于基础设施环境问题，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418193

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418330

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418366

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418444

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418483

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32328794292/job/96305418110) |


## [Run #32328523714](https://github.com/sgl-project/sglang/actions/runs/32328523714)
- **分支**: `lora-spec-multi-adapter`
- **总耗时**: 9.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32328523714

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642330) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642414) |
| base-b-test-2-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642460) |
| base-b-test-1-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642513) |
| base-b-test-4-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642537) |
| base-b-test-4-npu-a3 / run (1) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642552) |
| base-b-test-16-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642633) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642707) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642709) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642716) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642746) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642330

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642414

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642460

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642513

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642537

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642552

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642633

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642707

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642709

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642716

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642746

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32328523714/job/96304642378) |


## [Run #32328300726](https://github.com/sgl-project/sglang/actions/runs/32328300726)
- **分支**: `codex/longcat-docs-api`
- **总耗时**: 107.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32328300726

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328300726/job/96304021892) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328300726/job/96304021892


## [Run #32328300584](https://github.com/sgl-project/sglang/actions/runs/32328300584)
- **分支**: `fix/component-residency-report`
- **总耗时**: 98.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32328300584

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 54.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32328300584/job/96304013490) |

- **multimodal-gen-test-1-npu-a3**: 日志截断了中间测试执行部分，只显示开始和结束阶段。上传diffusion-failures工件时提示无文件，说明测试可能未产生失败样本，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32328300584/job/96304013490


## [Run #32327534719](https://github.com/sgl-project/sglang/actions/runs/32327534719)
- **分支**: `main`
- **总耗时**: 62.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32327534719

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 60.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859260) |
| base-b-test-4-npu-a3 / run (1) | 60.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859313) |
| base-b-test-1-npu-a3 / run (0) | 60.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859325) |
| base-b-test-16-npu-a3 / run (0) | 60.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859390) |
| base-b-test-8-npu-a3 / run (0) | 60.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859423) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 60.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859550) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 60.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859557) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 60.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859579) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 60.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859594) |
| base-b-test-4-npu-a3 / run (0) | 60.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859937) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859260

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859313

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859325

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859390

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859423

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859550

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的存储位置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859557

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859579

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859594

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理或路径配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859937

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32327534719/job/96301859290) |


## [Run #32327484201](https://github.com/sgl-project/sglang/actions/runs/32327484201)
- **分支**: `lora-spec-multi-adapter`
- **总耗时**: 17.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32327484201

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 17.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627368) |
| base-b-test-1-npu-a3 / run (0) | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627483) |
| base-b-test-4-npu-a3 / run (1) | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627522) |
| base-b-test-4-npu-a3 / run (0) | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627537) |
| base-b-test-2-npu-a3 / run (0) | 17.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627550) |
| base-b-test-16-npu-a3 / run (0) | 17.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627593) |
| base-b-test-8-npu-a3 / run (0) | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627637) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627692) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627708) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627780) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627813) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627368

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627483

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627522

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627537

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627550

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627593

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627637

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627692

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627708

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627780

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627813

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32327484201/job/96301627486) |


## [Run #32327427544](https://github.com/sgl-project/sglang/actions/runs/32327427544)
- **分支**: `fix/component-residency-report`
- **总耗时**: 14.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32327427544

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327427544/job/96301478859) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中不存在。这通常是因为日志或依赖文件未正确上传、被删除或路径错误，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327427544/job/96301478859


## [Run #32327311611](https://github.com/sgl-project/sglang/actions/runs/32327311611)
- **分支**: `codex/longcat-docs-api`
- **总耗时**: 16.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32327311611

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32327311611/job/96301132899) |

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置变更，属于环境或资源问题，需检查CI配置中的blob路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32327311611/job/96301132899


## [Run #32326719815](https://github.com/sgl-project/sglang/actions/runs/32326719815)
- **分支**: `dsv4_overlap`
- **总耗时**: 5.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32326719815

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449215) |
| base-b-test-2-npu-a3 / run (0) | 4.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449320) |
| base-b-test-8-npu-a3 / run (0) | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449334) |
| base-b-test-1-npu-a3 / run (0) | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449346) |
| base-a-test-1-npu-a2 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，NPU测试环境未正确启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449367) |
| base-b-test-16-npu-a3 / run (0) | 4.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449402) |
| base-b-test-4-npu-a3 / run (1) | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449444) |
| base-b-test-4-npu-a3 / run (0) | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449490) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449624) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449643) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449677) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449681) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被误删、路径配置错误或上传未完成，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449215

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、资源被清理或配置问题，属于环境或基础设施故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449320

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449334

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449346

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员。测试尚未开始即中断，属于环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449367

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449402

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449444

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449490

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449624

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449643

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449677

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326719815/job/96299449681


## [Run #32326018417](https://github.com/sgl-project/sglang/actions/runs/32326018417)
- **分支**: `dev`
- **总耗时**: 37.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32326018417

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 36.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374668) |
| base-b-test-8-npu-a3 / run (0) | 36.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374707) |
| base-b-test-16-npu-a3 / run (0) | 36.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374725) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | 代码错误 | NPU注意力测试中test_ascend_gdn_backend.py失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374738) |
| base-b-test-1-npu-a3 / run (0) | 36.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374758) |
| base-b-test-4-npu-a3 / run (1) | 36.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374790) |
| base-b-test-4-npu-a3 / run (0) | 36.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374851) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 36.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297375048) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297375084) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 36.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297375130) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 36.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297375132) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，可能是上游构建未成功上传或存储配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374668

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374707

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 存储中的 blob，但该 blob 不存在（BlobNotFound）。这通常是因为日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374725

- **base-a-test-1-npu-a2 / run (0)**: 测试test_ascend_gdn_backend.py执行4秒即退出，返回码1，导致整体作业失败。其余两个测试通过，问题定位在该测试文件本身，可能是代码逻辑或环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374738

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374758

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374790

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297374851

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储账户中找到，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297375048

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297375084

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297375130

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32326018417/job/96297375132


## [Run #32325597578](https://github.com/sgl-project/sglang/actions/runs/32325597578)
- **分支**: `lsyin/swa-free-alive`
- **总耗时**: 53.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32325597578

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 29.0min | 其他 | 作业日志被截断，未显示实际测试结果，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204379) |
| base-b-test-4-npu-a3 / run (1) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204457) |
| base-b-test-4-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204459) |
| base-b-test-2-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204462) |
| base-b-test-16-npu-a3 / run (0) | 52.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204474) |
| base-b-test-1-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204480) |
| base-b-test-8-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204545) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204686) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204734) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204742) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204786) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败原因。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204379

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204457

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是构建产物或依赖缓存缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204459

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志被清理、路径错误或上传失败，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204462

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204474

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204480

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查上传步骤或存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204545

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204686

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存、依赖或上传文件未正确生成，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204734

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204742

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204786

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 9.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32325597578/job/96296204486) |


## [Run #32325014239](https://github.com/sgl-project/sglang/actions/runs/32325014239)
- **分支**: `feat/aiter-verify-gqa-pack`
- **总耗时**: 200.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32325014239

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 200.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466247) |
| multimodal-gen-test-1-npu-a3 | 77.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466255) |
| base-b-test-8-npu-a3 / run (0) | 200.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466301) |
| base-b-test-4-npu-a3 / run (0) | 200.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466338) |
| base-b-test-4-npu-a3 / run (1) | 200.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466350) |
| base-b-test-2-npu-a3 / run (0) | 200.1min | 环境问题 | CI作业因Azure Blob存储中找不到指定文件而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466376) |
| base-b-test-16-npu-a3 / run (0) | 200.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466392) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 200.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466631) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 200.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466643) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 200.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466669) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 200.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466733) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466247

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact时未找到diffusion-failures目录的提示，未包含任何测试执行或失败的具体错误信息，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466255

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466301

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466338

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466350

- **base-b-test-2-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业依赖的某个文件（如模型权重或测试数据）在存储中不存在，可能是文件被误删、路径配置错误或上传未完成，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466376

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466392

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466631

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466643

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466669

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466733

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32325014239/job/96294466305) |


## [Run #32324979757](https://github.com/sgl-project/sglang/actions/runs/32324979757)
- **分支**: `fix/w4a8-ll-per-token-quant-k-tail`
- **总耗时**: 166.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32324979757

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 166.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357349) |
| base-b-test-4-npu-a3 / run (0) | 166.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357385) |
| base-b-test-8-npu-a3 / run (0) | 166.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357393) |
| base-b-test-1-npu-a3 / run (0) | 166.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357394) |
| base-b-test-16-npu-a3 / run (0) | 166.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357408) |
| base-b-test-4-npu-a3 / run (1) | 166.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357418) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 166.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357565) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 166.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357568) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 166.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357605) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 166.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357613) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357349

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357385

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查上传步骤或存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357393

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357394

- **base-b-test-16-npu-a3 / run (0)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志文件被清理、路径错误或上传失败，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357408

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357418

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357565

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357568

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357605

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查相关 blob 是否存在及访问权限。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357613

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32324979757/job/96294357313) |


## [Run #32324727239](https://github.com/sgl-project/sglang/actions/runs/32324727239)
- **分支**: `main`
- **总耗时**: 17.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32324727239

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700805) |
| base-b-test-8-npu-a3 / run (0) | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700844) |
| base-b-test-4-npu-a3 / run (0) | 16.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700878) |
| base-b-test-1-npu-a3 / run (0) | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700883) |
| base-b-test-16-npu-a3 / run (0) | 16.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700902) |
| base-b-test-2-npu-a3 / run (0) | 16.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700927) |
| base-b-test-4-npu-a3 / run (1) | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700936) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293701083) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293701093) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293701103) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293701131) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700805

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700844

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700878

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700883

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，属于基础设施或配置问题，需检查存储路径或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700902

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700927

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700936

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293701083

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293701093

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的预构建产物或缓存文件未上传或已被删除，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293701103

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293701131

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32324727239/job/96293700921) |


## [Run #32324696338](https://github.com/sgl-project/sglang/actions/runs/32324696338)
- **分支**: `feat/pd-role-switch-mori-v2`
- **总耗时**: 288.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32324696338

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 82.0min | 其他 | 作业日志被截断，未显示实际测试结果，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552660) |
| base-b-test-16-npu-a3 / run (0) | 288.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552711) |
| base-b-test-2-npu-a3 / run (0) | 288.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552802) |
| base-b-test-8-npu-a3 / run (0) | 288.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552808) |
| base-b-test-1-npu-a3 / run (0) | 288.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552815) |
| base-b-test-4-npu-a3 / run (1) | 288.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552880) |
| base-b-test-4-npu-a3 / run (0) | 288.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552884) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 288.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293553035) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 288.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293553166) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 288.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293553182) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 288.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293553227) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法判断测试是否失败。仅能看到上传diffusion-failures目录时提示无文件，说明该目录不存在或为空，可能测试未产生失败样本，或作业在测试阶段被中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552660

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552711

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传失败、路径错误或存储被清理所致，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552802

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552808

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552815

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552880

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552884

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293553035

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个远程资源（如模型权重、测试数据或缓存）已被删除或路径错误，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293553166

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293553182

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的某个文件（可能是模型权重、数据集或缓存）已被删除或路径错误，导致作业启动失败。需检查相关资源是否仍存在或路径配置是否正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293553227

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32324696338/job/96293552684) |


## [Run #32324615439](https://github.com/sgl-project/sglang/actions/runs/32324615439)
- **分支**: `codex/minimax-h3-ref2va-audio-ci`
- **总耗时**: 112.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32324615439

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 77.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32324615439/job/96293314126) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32324615439/job/96293314126


---
*Auto-generated by npu_pr_monitor.py*