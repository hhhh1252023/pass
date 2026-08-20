# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-20 23:37 UTC
**本次检查已合入 PR 数**: 35
**涉及 NPU**: 13 | **无关**: 22 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#35750](https://github.com/sgl-project/sglang/pull/35750) [CI] Gate `/rerun-test` on commenter trust and remove `/rerun-stage`
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#35412](https://github.com/sgl-project/sglang/pull/35412) [Fix] Land the decode mamba checkpoint depth on the tree page under DCP
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#34829](https://github.com/sgl-project/sglang/pull/34829) 📝 [NPU] Clean up quantization comments
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#35722](https://github.com/sgl-project/sglang/pull/35722) npu_transpose_batchmatmul ascendc
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#35716](https://github.com/sgl-project/sglang/pull/35716) Ifmn/npu/glm 5 optim origin dual stream
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#35293](https://github.com/sgl-project/sglang/pull/35293) test: switch the Inkling-Small NVFP4 deterministic suite to DSPARK
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#35699](https://github.com/sgl-project/sglang/pull/35699) 【NPU】memory optimize
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#34936](https://github.com/sgl-project/sglang/pull/34936) [NPU] [FIX] Fix non-contiguous parameter issue in FIA operator
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#34935](https://github.com/sgl-project/sglang/pull/34935) [NPU]Ensure tensors allocated by empty_like are contiguous
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#35418](https://github.com/sgl-project/sglang/pull/35418) [Diffusion] Support MiniMax-H3 pruned safetensors checkpoints
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#35641](https://github.com/sgl-project/sglang/pull/35641) [diffusion] feat: plan pinned host memory against the cgroup cap not the machine
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#35668](https://github.com/sgl-project/sglang/pull/35668) [diffusion] feat: add weight source reader
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#35511](https://github.com/sgl-project/sglang/pull/35511) [diffusion] CI: add minimax-h3 ref2va audio consistency coverage and guard peak vram
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 24

## ✅ 与 NPU 无关的已合入 PR
- [#32832](https://github.com/sgl-project/sglang/pull/32832) [AMD] [sgl-kernel] Bypass caches for peer traffic in ROCm custom all-reduce
- [#35323](https://github.com/sgl-project/sglang/pull/35323) fix(openai): avoid duplicate routed expert in response when `return_meta_info = True`
- [#35600](https://github.com/sgl-project/sglang/pull/35600) Add CI permissions for four contributors
- [#35756](https://github.com/sgl-project/sglang/pull/35756) [Docker] Defer CUDA 13 NCCL override until after dependency resolution
- [#35754](https://github.com/sgl-project/sglang/pull/35754) [Cherry-pick to release/v0.5.18] [Fix] Support 128-aligned hidden sizes in the W4AFP8 DeepEP low-latency requant kernel (#35593)
- [#35622](https://github.com/sgl-project/sglang/pull/35622) [misc] Trim restating comments and docstrings in srt/managers
- [#35554](https://github.com/sgl-project/sglang/pull/35554) [Kimi K3] Select FlashInfer MXFP4 for SM107 auto MoE
- [#35663](https://github.com/sgl-project/sglang/pull/35663) [docs] Add DFlash2 speculative cells to the Qwen3.8-27B cookbook
- [#35741](https://github.com/sgl-project/sglang/pull/35741) [Cherry-pick to release/v0.5.18] feat(grpc): expose KV event discovery metadata (#35714)
- [#35714](https://github.com/sgl-project/sglang/pull/35714) feat(grpc): expose KV event discovery metadata
- [#34406](https://github.com/sgl-project/sglang/pull/34406) TP/PP Consensus checker
- [#35700](https://github.com/sgl-project/sglang/pull/35700) fix(multimodal): keep LLaVA image fetch off the CPU-preprocess timeout budget (flaky test_mixed_batch)
- [#35689](https://github.com/sgl-project/sglang/pull/35689) Skip empty linear-attention state buffers in PD transfer
- [#35610](https://github.com/sgl-project/sglang/pull/35610) [MUSA] Harden CI dependencies and diffusion warmup
- [#35713](https://github.com/sgl-project/sglang/pull/35713) [diffusion] feat: support out-of-tree models and pipelines
- [#35688](https://github.com/sgl-project/sglang/pull/35688) [diffusion] feat: let every layerwise component be configurable
- [#35679](https://github.com/sgl-project/sglang/pull/35679) [diffusion] Refresh eager optimization skills and benchmark safeguards
- [#35632](https://github.com/sgl-project/sglang/pull/35632) [Fix] Keep deterministic GDN prefill on Triton
- [#35455](https://github.com/sgl-project/sglang/pull/35455) [Quant] Load compressed-tensors kv_cache_scheme scales
- [#35664](https://github.com/sgl-project/sglang/pull/35664) [diffusion] feat: warn on an unverified short edge instead of rejecting it for minimax-h3
- [#35603](https://github.com/sgl-project/sglang/pull/35603) [AMD][CI] Run Both ROCm 7.2.4 and ROCm 7.2.0 Images on Nightly Test AMD
- [#35602](https://github.com/sgl-project/sglang/pull/35602) [AMD][CI] Default the ROCm 7.2 PR gate to ROCm 7.2.4 Image

---
*Auto-generated by npu_pr_monitor.py*