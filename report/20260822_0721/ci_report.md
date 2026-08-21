# NPU CI 执行监控
**生成时间**: 2026-08-21 23:21 UTC
**分析 Run 数**: 28

---

## 📊 本次执行总结

- **成功 Job 数**: 56
- **失败 Run 数**: 28
- **成功 Job 平均耗时**: 22.3min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 101.9min | #32488929446 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878569) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.2min | #32484511326 | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110944) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.4min | #32484511326 | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96900495117) |
| base-b-test-16-npu-a3 / run (0) | 59.9min | #32484511326 | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110428) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 59.7min | #32484511326 | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110915) |
| base-b-test-16-npu-a3 / run (0) | 51.9min | #32488929446 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878106) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 45.0min | #32484511326 | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96901845773) |
| multimodal-gen-test-2-npu-a3 (0) | 41.0min | #32490189145 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925021) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.7min | #32488929446 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878488) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.8min | #32488929446 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878558) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 24 | 0 | 13 | 0 | 11 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 3 | 1 | 0 | 2 | 0 |
| base-b-test-16-npu-a3 / run (0) | 16 | 2 | 0 | 1 | 13 |
| base-a-test-1-npu-a2 / run (0) | 16 | 15 | 0 | 1 | 0 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 16 | 2 | 0 | 1 | 13 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 16 | 2 | 0 | 1 | 13 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 2 | 1 | 0 | 1 | 0 |
| multimodal-gen-test-2-npu-a3 | 1 | 0 | 1 | 0 | 0 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32484511326<br>[#35735 [kernel] Split the custom all-reduce communicator into push/pull planes](https://github.com/sgl-project/sglang/pull/35735) | `comm-plane-refactor` | 536.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32484511326) |
| #32488929446<br>[#34432 [AMD][DCP 1/N] add dcp support for aiter backend](https://github.com/sgl-project/sglang/pull/34432) | `k3_dcp_1n` | 530.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32488929446) |
| #32498735474<br>[#34938 perf: overlap Qwen shared expert with DeepEP routed experts](https://github.com/sgl-project/sglang/pull/34938) | `yangminl/agentx-decode-gap-v2-shared-overlap-v2-20260815` | 421.4min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32498735474) |
| #32493576936<br>[#34727 [kernel] One rmsnorm kernel for every hidden size, tuned from Python](https://github.com/sgl-project/sglang/pull/34727) | `jit-rmsnorm-copy-mode` | 217.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32493576936) |
| #32490189145<br>[#34855 [NPU] [Diffusion] Fix critical Ascend NPU Diffusion regression/bugs & restore 2-NPU CI testcase](https://github.com/sgl-project/sglang/pull/34855) | `fix_ring_attention_npu` | 106.9min | 10 | base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32490189145) |
| #32492726287 | `amd/spec-topk1-argmax-rocm` | 85.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32492726287) |
| #32500675945<br>[#35734 [diffusion] park a layerwise component's non-layer weights between uses](https://github.com/sgl-project/sglang/pull/35734) | `feat/park-non-layer-weights` | 82.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32500675945) |
| #32500819535<br>[#35882 [diffusion] ship mapped layers through a courier thread](https://github.com/sgl-project/sglang/pull/35882) | `feat/mapped-layer-courier` | 81.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32500819535) |
| #32496891330<br>[#35873 [diffusion] Fail closed for unsupported quantized component checkpoints](https://github.com/sgl-project/sglang/pull/35873) | `codex/diffusion-quantized-component-admission` | 79.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32496891330) |
| #32484697281<br>[#35745 [multimodal] MiniMax-H3: do not warn that the recommended short edge is unverified](https://github.com/sgl-project/sglang/pull/35745) | `fix/minimax-h3-short-edge-warning` | 79.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32484697281) |
| #32482481587<br>[#34855 [NPU] [Diffusion] Fix critical Ascend NPU Diffusion regression/bugs & restore 2-NPU CI testcase](https://github.com/sgl-project/sglang/pull/34855) | `fix_ring_attention_npu` | 77.0min | 11 | multimodal-gen-test-2-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32482481587) |
| #32490509986<br>[#35867 [diffusion] hand out pinned host memory per layer, and never oversell the host](https://github.com/sgl-project/sglang/pull/35867) | `feat/h3-partial-layer-pinning` | 75.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32490509986) |
| #32488408564<br>[#35796 [diffusion] fall back to a component's default attention backend](https://github.com/sgl-project/sglang/pull/35796) | `fix/diffusion-attention-backend-fallback` | 73.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32488408564) |
| #32486360563<br>[#35734 [diffusion] park a layerwise component's non-layer weights between uses](https://github.com/sgl-project/sglang/pull/35734) | `feat/park-non-layer-weights` | 68.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32486360563) |
| #32485089646<br>[#35774 [diffusion] Resolve LoRA weight sources deterministically](https://github.com/sgl-project/sglang/pull/35774) | `main` | 68.9min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32485089646) |
| #32485591271<br>[#35812 [diffusion] let the auto policy select H3's DiT for layerwise offload](https://github.com/sgl-project/sglang/pull/35812) | `fix/h3-auto-selects-dit` | 68.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32485591271) |
| #32487950355<br>[#35868 [diffusion] Preserve PEFT LoRA semantics](https://github.com/sgl-project/sglang/pull/35868) | `codex/diffusion-peft-lora-compact` | 68.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32487950355) |
| #32486267603<br>[#35867 [diffusion] hand out pinned host memory per layer, and never oversell the host](https://github.com/sgl-project/sglang/pull/35867) | `feat/h3-partial-layer-pinning` | 48.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32486267603) |
| #32490807077<br>[#35847 refactor(disagg): collapse duplicated branches in get_kv_class](https://github.com/sgl-project/sglang/pull/35847) | `main` | 44.9min | 11 | base-b-test-8-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32490807077) |
| #32488642649<br>[#35747 Add sampling observer auxiliary output hooks](https://github.com/sgl-project/sglang/pull/35747) | `alecs/sampling-observer-auxiliary-output` | 43.4min | 11 | base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32488642649) |
| #32487136935 | `kewen/dllm-indel-algorithm` | 42.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32487136935) |
| #32495102787<br>[#35868 [diffusion] Preserve PEFT LoRA semantics](https://github.com/sgl-project/sglang/pull/35868) | `main` | 31.4min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32495102787) |
| #32491494306<br>[#34727 [kernel] One rmsnorm kernel for every hidden size, tuned from Python](https://github.com/sgl-project/sglang/pull/34727) | `jit-rmsnorm-copy-mode` | 18.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32491494306) |
| #32499854192<br>[#34855 [NPU] [Diffusion] Fix critical Ascend NPU Diffusion regression/bugs & restore 2-NPU CI testcase](https://github.com/sgl-project/sglang/pull/34855) | `fix_ring_attention_npu` | 12.3min | 14 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32499854192) |
| #32500418546 | `amd/spec-topk1-argmax-rocm` | 9.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32500418546) |
| #32488979001<br>[#34855 [NPU] [Diffusion] Fix critical Ascend NPU Diffusion regression/bugs & restore 2-NPU CI testcase](https://github.com/sgl-project/sglang/pull/34855) | `fix_ring_attention_npu` | 8.1min | 14 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (1), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), multimodal-gen-test-2-npu-a3 (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32488979001) |
| #32490928064 | `jit-rmsnorm-copy-mode` | 6.4min | 11 | base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32490928064) |
| #32493128135 | `jit-rmsnorm-copy-mode` | 5.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32493128135) |

---


## [Run #32500819535](https://github.com/sgl-project/sglang/actions/runs/32500819535)
- **分支**: `feat/mapped-layer-courier`
- **总耗时**: 81.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32500819535

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500819535/job/96829923155) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件（if-no-files-found: ignore），以及Node 20弃用警告。实际失败原因可能被截断或未记录。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500819535/job/96829923155


## [Run #32500675945](https://github.com/sgl-project/sglang/actions/runs/32500675945)
- **分支**: `feat/park-non-layer-weights`
- **总耗时**: 82.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32500675945

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500675945/job/96829597232) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500675945/job/96829597232

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32500675945/job/96829372410) |


## [Run #32500418546](https://github.com/sgl-project/sglang/actions/runs/32500418546)
- **分支**: `amd/spec-topk1-argmax-rocm`
- **总耗时**: 9.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32500418546

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828707713) |
| base-b-test-16-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828707985) |
| base-b-test-1-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708066) |
| base-b-test-2-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708112) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708132) |
| base-b-test-4-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708173) |
| base-b-test-4-npu-a3 / run (1) | 8.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708338) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708597) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708686) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708702) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708742) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828707713

- **base-b-test-16-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828707985

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708066

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或过期清理所致，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708112

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708132

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708173

- **base-b-test-4-npu-a3 / run (1)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708338

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708597

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708686

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708702

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708742

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32500418546/job/96828708249) |


## [Run #32499854192](https://github.com/sgl-project/sglang/actions/runs/32499854192)
- **分支**: `fix_ring_attention_npu`
- **总耗时**: 12.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32499854192

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 7.2min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908025) |
| multimodal-gen-test-2-npu-a3 (0) | 11.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908133) |
| multimodal-gen-test-2-npu-a3 (1) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908185) |
| multimodal-gen-test-1-npu-a3 (1) | 11.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908416) |
| base-b-test-1-npu-a3 / run (0) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908606) |
| base-b-test-4-npu-a3 / run (0) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908720) |
| base-b-test-4-npu-a3 / run (1) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908731) |
| base-b-test-16-npu-a3 / run (0) | 11.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908745) |
| base-b-test-8-npu-a3 / run (0) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908826) |
| base-b-test-2-npu-a3 / run (0) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908946) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826909484) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826909545) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826909602) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826909604) |

- **multimodal-gen-test-1-npu-a3 (0)**: 日志中未出现测试失败或错误信息，仅有Node 20弃用警告及diffusion-failures目录无文件上传提示，可能为作业提前结束或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908025

- **multimodal-gen-test-2-npu-a3 (0)**: 作业失败原因是访问Azure Blob存储时返回BlobNotFound错误，即请求的资源不存在。这通常是由于日志或依赖文件被清理、路径错误或上传失败导致，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908133

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是临时文件被清理或配置问题，属于环境依赖故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908185

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI依赖的远程资源缺失或路径错误，可能是上传失败、文件被清理或配置指向了不存在的存储位置，需检查相关资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908416

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908606

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908720

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908731

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908745

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908826

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908946

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826909484

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826909545

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826909602

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826909604

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32499854192/job/96826908814) |


## [Run #32498735474](https://github.com/sgl-project/sglang/actions/runs/32498735474)
- **分支**: `yangminl/agentx-decode-gap-v2-shared-overlap-v2-20260815`
- **总耗时**: 421.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32498735474

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.4min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442599) |
| base-b-test-16-npu-a3 / run (0) | 34.4min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442999) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 33.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823443429) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 34.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823443545) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 8.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96929400042) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node.js版本弃用警告和上传artifact时未找到diffusion-failures目录的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442599

- **base-b-test-16-npu-a3 / run (0)**: 测试运行到第4个文件时，自定义容器实现执行失败，导致作业终止。日志显示测试前3个文件均通过，但在开始第4个测试时容器异常退出，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442999

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823443429

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在22:39:58出现错误'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823443545

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示模型权重加载到95%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96929400042

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442758) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442816) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442830) |
| base-b-test-1-npu-a3 / run (0) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442888) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823442943) |
| base-b-test-4-npu-a3 / run (1) | 10.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823443020) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823443467) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96823443491) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32498735474/job/96924986660) |


## [Run #32496891330](https://github.com/sgl-project/sglang/actions/runs/32496891330)
- **分支**: `codex/diffusion-quantized-component-admission`
- **总耗时**: 79.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32496891330

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 65.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32496891330/job/96817439829) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但无法从现有日志判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32496891330/job/96817439829


## [Run #32495102787](https://github.com/sgl-project/sglang/actions/runs/32495102787)
- **分支**: `main`
- **总耗时**: 31.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32495102787

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 28.9min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32495102787/job/96811671882) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions的初始化、Node版本警告和上传artifact步骤，未包含实际测试命令或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32495102787/job/96811671882


## [Run #32493576936](https://github.com/sgl-project/sglang/actions/runs/32493576936)
- **分支**: `jit-rmsnorm-copy-mode`
- **总耗时**: 217.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32493576936

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.9min | 环境问题 | 作业因缺少diffusion-failures目录而失败，但核心测试可能已通过或未执行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980319) |
| base-b-test-2-npu-a3 / run (0) | 215.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980558) |
| base-b-test-8-npu-a3 / run (0) | 215.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980681) |
| base-b-test-1-npu-a3 / run (0) | 215.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980687) |
| base-b-test-16-npu-a3 / run (0) | 215.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980745) |
| base-b-test-4-npu-a3 / run (0) | 215.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980756) |
| base-b-test-4-npu-a3 / run (1) | 215.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980831) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 215.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806981648) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 215.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806981737) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 215.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806981856) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 215.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806981865) |

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤未找到diffusion-failures/文件，说明测试未产生失败样本。作业可能因测试未运行或环境配置问题导致提前结束，而非实际测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980319

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980558

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980681

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980687

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查上传步骤或存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980745

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境或配置问题，需检查相关存储路径或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980756

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980831

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806981648

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806981737

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806981856

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806981865

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 9.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32493576936/job/96806980481) |


## [Run #32493128135](https://github.com/sgl-project/sglang/actions/runs/32493128135)
- **分支**: `jit-rmsnorm-copy-mode`
- **总耗时**: 5.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32493128135

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 1.2min | 环境问题 | 作业在准备阶段因Node.js 20弃用警告被强制使用Node 24运行，但未导致直接失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805395727) |
| base-b-test-2-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805395898) |
| base-b-test-16-npu-a3 / run (0) | 4.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805395950) |
| base-b-test-4-npu-a3 / run (1) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396063) |
| base-b-test-4-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396188) |
| base-b-test-1-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396218) |
| base-b-test-8-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396239) |
| base-a-test-1-npu-a2 / run (0) | 4.1min | 环境问题 | 自定义容器执行失败，NPU测试环境未正常启动 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396303) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396960) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805397079) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805397206) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805397272) |

- **multimodal-gen-test-1-npu-a3**: 日志显示actions/checkout和upload-artifact因Node 20弃用被强制在Node 24上运行，产生警告，但作业尚未进入实际测试阶段即结束，可能因环境配置或资源调度问题中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805395727

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805395898

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储账户中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805395950

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396063

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396188

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储文件缺失，可能是构建产物或依赖未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396218

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396239

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动自定义容器时失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396303

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805396960

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805397079

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805397206

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32493128135/job/96805397272


## [Run #32492726287](https://github.com/sgl-project/sglang/actions/runs/32492726287)
- **分支**: `amd/spec-topk1-argmax-rocm`
- **总耗时**: 85.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32492726287

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038725) |
| base-b-test-2-npu-a3 / run (0) | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038750) |
| base-b-test-8-npu-a3 / run (0) | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038839) |
| base-b-test-1-npu-a3 / run (0) | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038899) |
| base-b-test-4-npu-a3 / run (1) | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038901) |
| base-b-test-16-npu-a3 / run (0) | 84.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038906) |
| base-b-test-4-npu-a3 / run (0) | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038991) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804039202) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804039255) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804039336) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 84.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804039400) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038725

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038750

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是上游产物未上传或过期，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038839

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038899

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038901

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是日志上传或依赖下载路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038906

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038991

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804039202

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804039255

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804039336

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804039400

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32492726287/job/96804038843) |


## [Run #32491494306](https://github.com/sgl-project/sglang/actions/runs/32491494306)
- **分支**: `jit-rmsnorm-copy-mode`
- **总耗时**: 18.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32491494306

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800124861) |
| base-b-test-16-npu-a3 / run (0) | 17.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125093) |
| base-b-test-8-npu-a3 / run (0) | 17.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125125) |
| base-b-test-1-npu-a3 / run (0) | 17.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125160) |
| base-b-test-4-npu-a3 / run (1) | 17.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125185) |
| base-b-test-2-npu-a3 / run (0) | 17.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125231) |
| base-b-test-4-npu-a3 / run (0) | 17.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125295) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125435) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125445) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 17.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125451) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 17.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125488) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800124861

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125093

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象已被删除或路径错误，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125125

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125160

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125185

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施/环境配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125231

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125295

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125435

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125445

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125451

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125488

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32491494306/job/96800125078) |


## [Run #32490928064](https://github.com/sgl-project/sglang/actions/runs/32490928064)
- **分支**: `jit-rmsnorm-copy-mode`
- **总耗时**: 6.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32490928064

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238011) |
| multimodal-gen-test-1-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238045) |
| base-b-test-4-npu-a3 / run (1) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238077) |
| base-b-test-8-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238090) |
| base-b-test-2-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238096) |
| base-b-test-4-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238147) |
| base-b-test-16-npu-a3 / run (0) | 5.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238266) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238505) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238507) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238528) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238652) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238011

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238045

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238077

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238090

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238096

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238147

- **base-b-test-16-npu-a3 / run (0)**: 作业在尝试访问Azure Blob存储时，返回BlobNotFound错误，说明所需文件或资源未找到，可能是CI配置中引用的路径或文件名错误，或存储内容被清理/未上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238266

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238505

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238507

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238528

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238652

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32490928064/job/96798238146) |


## [Run #32490807077](https://github.com/sgl-project/sglang/actions/runs/32490807077)
- **分支**: `main`
- **总耗时**: 44.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32490807077

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 41.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706545) |
| multimodal-gen-test-1-npu-a3 | 35.1min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706562) |
| base-b-test-4-npu-a3 / run (1) | 41.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706667) |
| base-b-test-1-npu-a3 / run (0) | 41.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706692) |
| base-b-test-16-npu-a3 / run (0) | 41.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706783) |
| base-b-test-4-npu-a3 / run (0) | 41.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706808) |
| base-b-test-2-npu-a3 / run (0) | 41.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706825) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 41.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798707120) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 41.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798707125) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 41.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798707194) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798707227) |

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706545

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures目录时未找到文件，但未出现测试失败或错误信息，可能测试通过或日志不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706562

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706667

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706692

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是由于文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706783

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706808

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706825

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798707120

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798707125

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798707194

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798707227

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32490807077/job/96798706605) |


## [Run #32490509986](https://github.com/sgl-project/sglang/actions/runs/32490509986)
- **分支**: `feat/h3-partial-layer-pinning`
- **总耗时**: 75.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32490509986

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490509986/job/96796964224) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490509986/job/96796964224


## [Run #32490189145](https://github.com/sgl-project/sglang/actions/runs/32490189145)
- **分支**: `fix_ring_attention_npu`
- **总耗时**: 106.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32490189145

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925172) |
| base-b-test-2-npu-a3 / run (0) | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925182) |
| base-b-test-8-npu-a3 / run (0) | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925190) |
| base-b-test-1-npu-a3 / run (0) | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925251) |
| base-b-test-4-npu-a3 / run (0) | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925325) |
| base-b-test-16-npu-a3 / run (0) | 106.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925354) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925699) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925768) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925795) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 106.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925854) |

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925172

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925182

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925190

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传、被删除或配置有误，导致作业启动失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925251

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925325

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、资源被清理或配置问题，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925354

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925699

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925768

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925795

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925854

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795924832) |
| multimodal-gen-test-1-npu-a3 (1) | 13.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795924870) |
| multimodal-gen-test-2-npu-a3 (0) | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925021) |
| multimodal-gen-test-2-npu-a3 (1) | 33.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925068) |
| base-a-test-1-npu-a2 / run (0) | 8.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32490189145/job/96795925099) |


## [Run #32488979001](https://github.com/sgl-project/sglang/actions/runs/32488979001)
- **分支**: `fix_ring_attention_npu`
- **总耗时**: 8.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32488979001

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 3.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792017852) |
| multimodal-gen-test-1-npu-a3 (1) | 2.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792017887) |
| multimodal-gen-test-2-npu-a3 (1) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018057) |
| base-b-test-1-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018064) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018110) |
| base-b-test-16-npu-a3 / run (0) | 7.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018120) |
| multimodal-gen-test-2-npu-a3 (0) | 7.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018129) |
| base-b-test-2-npu-a3 / run (0) | 7.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018240) |
| base-b-test-4-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018277) |
| base-b-test-4-npu-a3 / run (1) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018357) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018691) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018773) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018783) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018794) |

- **multimodal-gen-test-1-npu-a3 (0)**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未包含任何测试执行或失败的具体内容，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792017852

- **multimodal-gen-test-1-npu-a3 (1)**: 日志仅显示runner启动、actions下载、上传artifact（无文件）及清理过程，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792017887

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018057

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018064

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018110

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试访问的日志或工件文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018120

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明作业依赖的某个文件或数据在存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018129

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018240

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018277

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的 blob。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018357

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018691

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018773

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018783

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018794

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488979001/job/96792018349) |


## [Run #32488929446](https://github.com/sgl-project/sglang/actions/runs/32488929446)
- **分支**: `k3_dcp_1n`
- **总耗时**: 530.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32488929446

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.7min | 其他 | 作业日志不完整，未显示测试执行过程，仅见上传产物时无文件，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791877906) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 35.2min | 性能回归 | NPU性能测试中qwen3_235b测试失败，可能因性能未达标 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96919269480) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96930484543) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions基础设施信息（Node版本警告、上传artifact时无diffusion-failures文件），未包含实际测试命令输出或错误信息，可能因日志截断或作业在测试阶段前已失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791877906

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件3个用例中1个失败，qwen3_235b_w8a8_8p_in3k5_out1k5_50ms测试退出码1，耗时1252秒，疑似性能指标未达预期或运行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96919269480

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-16-npu-a3 作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96930484543

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878058) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878071) |
| base-b-test-4-npu-a3 / run (0) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878094) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878102) |
| base-b-test-16-npu-a3 / run (0) | 51.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878106) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878134) |
| base-b-test-8-npu-a3 / run (0) | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878135) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878488) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878524) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878558) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 101.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96791878569) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96907597060) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488929446/job/96919215826) |


## [Run #32488642649](https://github.com/sgl-project/sglang/actions/runs/32488642649)
- **分支**: `alecs/sampling-observer-auxiliary-output`
- **总耗时**: 43.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32488642649

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968232) |
| base-b-test-1-npu-a3 / run (0) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968233) |
| multimodal-gen-test-1-npu-a3 | 37.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968237) |
| base-b-test-2-npu-a3 / run (0) | 42.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968245) |
| base-b-test-16-npu-a3 / run (0) | 42.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968312) |
| base-b-test-4-npu-a3 / run (0) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968348) |
| base-b-test-8-npu-a3 / run (0) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968406) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968577) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968637) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968659) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968775) |

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968232

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968233

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968237

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/日志文件在存储中缺失，可能是由于上游作业未成功上传或存储配置错误，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968245

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968312

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968348

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968406

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968577

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968637

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到。这通常是因为日志或依赖文件被删除、路径错误或上传失败，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968659

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968775

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32488642649/job/96790968285) |


## [Run #32488408564](https://github.com/sgl-project/sglang/actions/runs/32488408564)
- **分支**: `fix/diffusion-attention-backend-fallback`
- **总耗时**: 73.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32488408564

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 60.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32488408564/job/96790202897) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures工件时未找到文件，可能测试未运行或提前结束，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32488408564/job/96790202897


## [Run #32487950355](https://github.com/sgl-project/sglang/actions/runs/32487950355)
- **分支**: `codex/diffusion-peft-lora-compact`
- **总耗时**: 68.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32487950355

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.7min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487950355/job/96791470956) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或日志被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487950355/job/96791470956


## [Run #32487136935](https://github.com/sgl-project/sglang/actions/runs/32487136935)
- **分支**: `kewen/dllm-indel-algorithm`
- **总耗时**: 42.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32487136935

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 27.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286810) |
| base-b-test-8-npu-a3 / run (0) | 41.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286831) |
| base-b-test-4-npu-a3 / run (1) | 41.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286836) |
| base-b-test-4-npu-a3 / run (0) | 41.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286866) |
| base-b-test-2-npu-a3 / run (0) | 41.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286897) |
| base-b-test-16-npu-a3 / run (0) | 41.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286928) |
| base-b-test-1-npu-a3 / run (0) | 41.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286942) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786287368) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786287371) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786287382) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786287422) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传工件步骤，且未找到diffusion-failures文件，说明测试可能未执行或已通过，但缺少关键测试输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286810

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286831

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286836

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286866

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286897

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是日志上传或下载路径配置错误，或存储内容被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286928

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286942

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786287368

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786287371

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786287382

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786287422

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32487136935/job/96786286918) |


## [Run #32486360563](https://github.com/sgl-project/sglang/actions/runs/32486360563)
- **分支**: `feat/park-non-layer-weights`
- **总耗时**: 68.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32486360563

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32486360563/job/96783790012) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js弃用警告和上传artifact时未找到文件，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32486360563/job/96783790012


## [Run #32486267603](https://github.com/sgl-project/sglang/actions/runs/32486267603)
- **分支**: `feat/h3-partial-layer-pinning`
- **总耗时**: 48.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32486267603

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.5min | 其他 | 作业日志不完整，未显示测试执行过程，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32486267603/job/96783482809) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含实际测试命令或错误输出，仅显示上传diffusion-failures目录时无文件，可能测试未运行或提前退出，需查看完整日志定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32486267603/job/96783482809


## [Run #32485591271](https://github.com/sgl-project/sglang/actions/runs/32485591271)
- **分支**: `fix/h3-auto-selects-dit`
- **总耗时**: 68.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32485591271

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.4min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传工件提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32485591271/job/96781366013) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅有Node 20弃用警告和diffusion-failures目录无文件上传的提示，可能作业因其他原因被取消或提前结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/32485591271/job/96781366013


## [Run #32485089646](https://github.com/sgl-project/sglang/actions/runs/32485089646)
- **分支**: `main`
- **总耗时**: 68.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32485089646

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 59.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32485089646/job/96779826512) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node版本弃用警告和artifact上传提示（无文件）。可能因日志截断或作业在测试前被取消，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32485089646/job/96779826512


## [Run #32484697281](https://github.com/sgl-project/sglang/actions/runs/32484697281)
- **分支**: `fix/minimax-h3-short-edge-warning`
- **总耗时**: 79.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32484697281

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32484697281/job/96778604116) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32484697281/job/96778604116


## [Run #32484511326](https://github.com/sgl-project/sglang/actions/runs/32484511326)
- **分支**: `comm-plane-refactor`
- **总耗时**: 536.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32484511326

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传diffusion-failures目录时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110296) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最终上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110296

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110332) |
| base-b-test-16-npu-a3 / run (0) | 59.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110428) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110463) |
| base-b-test-4-npu-a3 / run (0) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110465) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110540) |
| base-b-test-4-npu-a3 / run (1) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110556) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110638) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110905) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 59.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110915) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110941) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96778110944) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96882055928) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96889408930) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96900495117) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 45.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32484511326/job/96901845773) |


## [Run #32482481587](https://github.com/sgl-project/sglang/actions/runs/32482481587)
- **分支**: `fix_ring_attention_npu`
- **总耗时**: 77.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32482481587

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 41.5min | 精度回归 | 多模态生成测试失败，上传了diffusion-failures工件，表明生成结果与预期不符。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771824864) |
| base-b-test-16-npu-a3 / run (0) | 76.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771824933) |
| base-b-test-4-npu-a3 / run (1) | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825005) |
| base-b-test-8-npu-a3 / run (0) | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825018) |
| base-b-test-1-npu-a3 / run (0) | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825055) |
| base-b-test-4-npu-a3 / run (0) | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825059) |
| base-b-test-2-npu-a3 / run (0) | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825086) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825503) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825517) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825578) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 76.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825647) |

- **multimodal-gen-test-2-npu-a3**: 作业运行约40分钟后上传了diffusion-failures-npu-2-1.zip工件，包含失败样本，说明多模态生成测试出现精度回归，需检查diffusion模型输出差异。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771824864

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771824933

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825005

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825018

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825055

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825059

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查 blob 名称和存储账户。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825086

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的 Azure Blob 存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825503

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825517

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825578

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825647

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 (0) | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771824914) |
| multimodal-gen-test-1-npu-a3 (1) | 12.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771824948) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32482481587/job/96771825061) |


---
*Auto-generated by npu_pr_monitor.py*