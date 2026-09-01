# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-01 09:43 UTC
**本次检查已合入 PR 数**: 33
**涉及 NPU**: 14 | **无关**: 19 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#33237](https://github.com/sgl-project/sglang/pull/33237) [FlashInfer V0.6.18] feat(dsv4): support --dsa-topk-backend flashinfer with fused top-k
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#36721](https://github.com/sgl-project/sglang/pull/36721) [mem_cache] Add `free_kv_row` to release a request's kv row by row range
- **检测方式**: 关键词初筛 + AI确认
- **理由**: 修改了streaming_session.py中NPU相关条件分支，涉及NPU页面对齐逻辑。
- **文件数**: 12

### [#37162](https://github.com/sgl-project/sglang/pull/37162) [Diffusion] Fuse FLUX.2 ModelOpt FP8 producers and QKV packing
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 16

### [#37047](https://github.com/sgl-project/sglang/pull/37047) fix(vlm): contain multimodal feature transport failures
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 16

### [#33926](https://github.com/sgl-project/sglang/pull/33926) [DCP] Support decode context parallelism on the trtllm_mla decode path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#35500](https://github.com/sgl-project/sglang/pull/35500) [CI/NPU] Isolate multi-node tests by run_id to prevent concurrent-run…
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#37307](https://github.com/sgl-project/sglang/pull/37307) fix(unified-memory): forward the KV-index translator through every wrapper backend
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#36624](https://github.com/sgl-project/sglang/pull/36624) [Cohere Command-A-Plus] Optimize decode and BCG capture on SM10X
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#35703](https://github.com/sgl-project/sglang/pull/35703) [diffusion] fix: fix loading a block-FP8 quantized MiniMax-H3 DiT
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36680](https://github.com/sgl-project/sglang/pull/36680) [Diffusion] Optimize Qwen-Image TP collectives and attention
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 19

### [#36699](https://github.com/sgl-project/sglang/pull/36699) xpu: record per-model metrics to jsonl for nightly dashboard
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#35120](https://github.com/sgl-project/sglang/pull/35120) [FlashInfer v0.6.18] add FlashInfer CuTe DSL NVFP4 W4A16 mode
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#37112](https://github.com/sgl-project/sglang/pull/37112) [Diffusion] Fuse FLUX.2 gated residual normalization on Blackwell
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#37123](https://github.com/sgl-project/sglang/pull/37123) [Diffusion] Fuse Qwen-Image FP8 QKV projection and Blackwell epilogue
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

## ✅ 与 NPU 无关的已合入 PR
- [#37380](https://github.com/sgl-project/sglang/pull/37380) Revert "[AMD] Add GLM-5.3-Flash recipes for MI300X, MI325X, and MI355X (#36608)"
- [#37366](https://github.com/sgl-project/sglang/pull/37366) Fix GLM-5.3 Flash CI regressions after latest main merge
- [#37351](https://github.com/sgl-project/sglang/pull/37351) [Cookbook] Add NVFP4 options for DeepSeek-V4 Flash Official (0731) and Pro Official (0813)
- [#37374](https://github.com/sgl-project/sglang/pull/37374) [CI] Fix hybrid wrapper test fake missing kv_index_translator
- [#34647](https://github.com/sgl-project/sglang/pull/34647) [AMD] Enable 12-head MLA aiter fp8 Gluon decode (batched bh16bn128).
- [#37338](https://github.com/sgl-project/sglang/pull/37338) [Fix][CPU] fix xeon ci failure by test_qwen35_flashinfer_fusion
- [#37109](https://github.com/sgl-project/sglang/pull/37109) [Docs] Add NVFP4 section to GLM-5.3-Flash cookbook
- [#37299](https://github.com/sgl-project/sglang/pull/37299) refactor(hicache): simplify decode offload state bookkeeping
- [#37298](https://github.com/sgl-project/sglang/pull/37298) fix: resolve CI regressions after GLM-5.3 Flash rebase
- [#37345](https://github.com/sgl-project/sglang/pull/37345) test: update hybrid attention runner fixtures
- [#37286](https://github.com/sgl-project/sglang/pull/37286) [AMD][MORI] Bump MoRI to 7c51d18 for ionic RoCE dmabuf fix (#509)
- [#37205](https://github.com/sgl-project/sglang/pull/37205) [Unified Cache][4/N]: Add Mooncake backend for external linker
- [#34967](https://github.com/sgl-project/sglang/pull/34967) [MoE] Add FlashInfer SM90 MXFP4 W4A8 CUTLASS MoE
- [#37339](https://github.com/sgl-project/sglang/pull/37339) [Fix] Use real ReqKvInfo in unit-test req mocks
- [#30915](https://github.com/sgl-project/sglang/pull/30915) [Feature] Megatron LayerNorm sequence parallelism (--enable-layernorm-sp)
- [#37317](https://github.com/sgl-project/sglang/pull/37317) [Kernel] Raise shape limits in shared FLA and MoE kernels (ported from #36507)
- [#37129](https://github.com/sgl-project/sglang/pull/37129) [Diffusion] Fuse Qwen-Image residual norm and NVFP4 quantization
- [#37279](https://github.com/sgl-project/sglang/pull/37279) Bump sgl-deep-gemm to 0.1.7
- [#37301](https://github.com/sgl-project/sglang/pull/37301) [Cookbook] Enable DSpark on the DeepSeek-V4 Flash Vision low-latency recipes

---
*Auto-generated by npu_pr_monitor.py*