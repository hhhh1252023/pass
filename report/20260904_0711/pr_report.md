# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-03 23:11 UTC
**本次检查已合入 PR 数**: 48
**涉及 NPU**: 13 | **无关**: 35 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#37629](https://github.com/sgl-project/sglang/pull/37629) [AMD] [GLM-5.3-Flash Day 0] Enable FP8 and Quark MXFP4 MoE on gfx950
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#37552](https://github.com/sgl-project/sglang/pull/37552) [1/N] Quantization Refactor: remove dead code and dedup the FP4 marlin helpers
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#37859](https://github.com/sgl-project/sglang/pull/37859) [Cherry-pick to release/v0.5.19] Fix GPU kernel ordering and MXFP8 quantization dispatch (#37331)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#37858](https://github.com/sgl-project/sglang/pull/37858) [Cherry-pick to release/v0.5.19] [ROCm][Bugfix] Cap the DSA MQA-logits budget at AITER's buffer_store limit (#36960)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#37855](https://github.com/sgl-project/sglang/pull/37855) [Cherry-pick to release/v0.5.19] [AMD] Gate the aiter memory-reserve exemption behind an env var (#37242)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#37854](https://github.com/sgl-project/sglang/pull/37854) [Cherry-pick to release/v0.5.19] Converge the two SWA predicates, and stop conditioning the capture sink on the pool (#37550)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#37464](https://github.com/sgl-project/sglang/pull/37464) [HiCache] buffer mode: anchor-lock staged prefetches by default
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#37266](https://github.com/sgl-project/sglang/pull/37266) [diffusion] MiniMax-H3: tiered AdaLN plan cache (pinned-host tier + per-plan LRU)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 15

### [#37332](https://github.com/sgl-project/sglang/pull/37332) [Diffusion][minimax-h3] Add SM120 support for SubBlock sparse attention
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#37760](https://github.com/sgl-project/sglang/pull/37760) [CI][NPU] Fix kimi_k2_6 16p in64k perf test and dsv4-flash testcases
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#37799](https://github.com/sgl-project/sglang/pull/37799) [NPU] [DOC] Refresh supported models and features on NPU
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#37693](https://github.com/sgl-project/sglang/pull/37693) [Feature] Unified memory: support decode context parallelism for the trtllm_mla family
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#33838](https://github.com/sgl-project/sglang/pull/33838) [AMD] Perf Kimi-K3 MoE optimization
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

## ✅ 与 NPU 无关的已合入 PR
- [#37881](https://github.com/sgl-project/sglang/pull/37881) [CI] Add Lark notifications for CUDA CI status, runner health, and queue time
- [#37567](https://github.com/sgl-project/sglang/pull/37567) Fix buffer-mode idle tracking and VLM memory sizing
- [#37454](https://github.com/sgl-project/sglang/pull/37454) [PD] Gate deferred decode KV release on backend capability
- [#37880](https://github.com/sgl-project/sglang/pull/37880) Revert "[AMD] [GLM-5.3-Flash Day 0] Enable FP8 and Quark MXFP4 MoE on gfx950"
- [#37665](https://github.com/sgl-project/sglang/pull/37665) [CI] Add dspark + dsv4 e2e test
- [#37867](https://github.com/sgl-project/sglang/pull/37867) [Cherry-pick to release/v0.5.19] [AMD][CI] Correct MI355X Slurm exclude node (#37779)
- [#37873](https://github.com/sgl-project/sglang/pull/37873) [Test] Allow top-k cutoff ties in `test_sampling_mask_matches_topk_logprobs`
- [#37866](https://github.com/sgl-project/sglang/pull/37866) [Cherry-pick to release/v0.5.19] [AMD][CI] Exclude unavailable MI355X nodes and skip 4N nightly (#37518)
- [#37865](https://github.com/sgl-project/sglang/pull/37865) [Cherry-pick to release/v0.5.19] [CI] Graceful teardown for the PD and HiSparse server fixtures (#37485)
- [#37864](https://github.com/sgl-project/sglang/pull/37864) [Cherry-pick to release/v0.5.19] [CI] Authenticate and retry git clones in install scripts (#37647)
- [#37863](https://github.com/sgl-project/sglang/pull/37863) [Cherry-pick to release/v0.5.19] [AMD] Fix nightly ROCm 7.0 image build: patch missing <optional> include in AITER topk kernel (#36216)
- [#37862](https://github.com/sgl-project/sglang/pull/37862) [Cherry-pick to release/v0.5.19] [AMD] Fix DSv4 draft extend taking the target compression path during prefill (#37713)
- [#37860](https://github.com/sgl-project/sglang/pull/37860) [Cherry-pick to release/v0.5.19] [ROCm] Define the DSA head-gate graph helpers on HIP (#37118)
- [#37857](https://github.com/sgl-project/sglang/pull/37857) [Cherry-pick to release/v0.5.19] [AMD] Fix v4 topk issue (#37439)
- [#37856](https://github.com/sgl-project/sglang/pull/37856) [Cherry-pick to release/v0.5.19] [AMD] fix aiter cannot get heuristic kernel regression (#37438)
- [#37853](https://github.com/sgl-project/sglang/pull/37853) [Cherry-pick to release/v0.5.19] [Fix] DP attention: correct the decode->extend prefix off-by-one (#37505)
- [#37729](https://github.com/sgl-project/sglang/pull/37729) [mem_cache] Require page-aligned starts in `free_segment` and drop the boundary trim
- [#37788](https://github.com/sgl-project/sglang/pull/37788) [Docs] [BugFix] Sync --tool-call-parser and --reasoning-parser lists with the code
- [#37502](https://github.com/sgl-project/sglang/pull/37502) [Scheduler] Count the parked chunked-prefill request in the busy mem check
- [#37737](https://github.com/sgl-project/sglang/pull/37737) [Cookbook] DeepSeek-V4 DGX Spark: v2 image + Flash Official NVFP4 and Flash Vision FP4 cells
- [#37381](https://github.com/sgl-project/sglang/pull/37381) [Unified Cache][5/N]: Integrate external linker mode end to end
- [#37849](https://github.com/sgl-project/sglang/pull/37849) Fix block-scale swizzling device placement
- [#36403](https://github.com/sgl-project/sglang/pull/36403) Support speculative decoding with unified SWA memory
- [#37844](https://github.com/sgl-project/sglang/pull/37844) [Cache] Forward fast prefix matching capability
- [#37731](https://github.com/sgl-project/sglang/pull/37731) [Router] Add composable scoring and eligibility policies
- [#37825](https://github.com/sgl-project/sglang/pull/37825) [Bugfix] Support K2 Horizon MoE without MoVA
- [#37616](https://github.com/sgl-project/sglang/pull/37616) [diffusion] loader: filter duplicate precision variants across custom loaders
- [#36735](https://github.com/sgl-project/sglang/pull/36735) [multimodal_gen] feat: support key masks on USPAttention's replicated-prefix path
- [#35922](https://github.com/sgl-project/sglang/pull/35922) [diffusion] feat: add maybe_record_function profiler spans for request phases
- [#37667](https://github.com/sgl-project/sglang/pull/37667) [Speculative Decoding] Add native UNO serving support
- [#37781](https://github.com/sgl-project/sglang/pull/37781) [AMD] Update kimi-k3 amd cookbook 0903
- [#37779](https://github.com/sgl-project/sglang/pull/37779) [AMD][CI] Correct MI355X Slurm exclude node
- [#37713](https://github.com/sgl-project/sglang/pull/37713) [AMD] Fix DSv4 draft extend taking the target compression path during prefill
- [#34187](https://github.com/sgl-project/sglang/pull/34187) [Kimi K3] Rework skipped-think fix as opt-in force_nonempty_content with streaming coverage
- [#37623](https://github.com/sgl-project/sglang/pull/37623) fix(benchmark): support Glm4MoeLite in fused MoE tuner

---
*Auto-generated by npu_pr_monitor.py*