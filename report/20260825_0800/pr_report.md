# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-25 00:00 UTC
**本次检查已合入 PR 数**: 44
**涉及 NPU**: 3 | **无关**: 41 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#30918](https://github.com/sgl-project/sglang/pull/30918) [Benchmark] Add optional steady-state window for serving metrics
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#35719](https://github.com/sgl-project/sglang/pull/35719) [AMD] Fix Qwen3.5 MTP dropping fused shared-expert weights
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#34066](https://github.com/sgl-project/sglang/pull/34066) perf(unified-memory): batch lazy-compaction mapping lookup
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

## ✅ 与 NPU 无关的已合入 PR
- [#20208](https://github.com/sgl-project/sglang/pull/20208) Remove maxItems=1 restriction when tool_choice is specified
- [#21181](https://github.com/sgl-project/sglang/pull/21181) [Spec][Ngram] 2/N: Rename branch length to max trie depth
- [#20393](https://github.com/sgl-project/sglang/pull/20393) [Spec][Ngram] 1/N: Reference based Speculative Decoding refactor
- [#21186](https://github.com/sgl-project/sglang/pull/21186) [Spec][Ngram] 3/N: Fix synchronization issues in `Ngram.cpp`
- [#22487](https://github.com/sgl-project/sglang/pull/22487) [Spec][Ngram] Clean up unused stateless `batchMatch`
- [#22294](https://github.com/sgl-project/sglang/pull/22294) [Spec][Ngram] Misc enhance support for multiple SAMs
- [#21243](https://github.com/sgl-project/sglang/pull/21243) [Spec][Ngram] 5/N: Store and advance anchor match state across decode steps
- [#19807](https://github.com/sgl-project/sglang/pull/19807) Fix issue 19717 by making `qo_indptr` uniform strided instead of packed
- [#21435](https://github.com/sgl-project/sglang/pull/21435) [Security] 1/N: Bind ZMQ sockets to localhost to prevent unauthenticated remote access
- [#21225](https://github.com/sgl-project/sglang/pull/21225) [Spec][Ngram] 4/N: Remove `max_match_window_size` and `min_match_window_size`, matching all suffixes of the Trie
- [#19899](https://github.com/sgl-project/sglang/pull/19899) [Spec] Refactor NaN/OOB checks to async `maybe_detect_*` with env-var control
- [#19819](https://github.com/sgl-project/sglang/pull/19819) Add kpham-sgl into CI Permission list
- [#21425](https://github.com/sgl-project/sglang/pull/21425) [Spec][Ngram] 6/N: Load an external corpus and construct a Suffix Automaton
- [#25547](https://github.com/sgl-project/sglang/pull/25547) Respect user override for Gemma4 attention backend
- [#20004](https://github.com/sgl-project/sglang/pull/20004) Multi tool streaming fix
- [#28601](https://github.com/sgl-project/sglang/pull/28601) [Fix] Return streaming logprobs when reasoning/tool parser is active
- [#27101](https://github.com/sgl-project/sglang/pull/27101) [Gemma4] Use hard GSM8K accuracy floor for 31B MTP test
- [#22471](https://github.com/sgl-project/sglang/pull/22471) [Spec][Ngram] Return token counts in list_external_corpora API
- [#25026](https://github.com/sgl-project/sglang/pull/25026) [Bench] Add MEM profile activity to bench_serving
- [#36025](https://github.com/sgl-project/sglang/pull/36025) [AMD][MORI] Deduplicate CP-replicated state transfers
- [#36204](https://github.com/sgl-project/sglang/pull/36204) docs: mark Ling-3.0-flash DSPARK verified for all four quantizations on H200
- [#36020](https://github.com/sgl-project/sglang/pull/36020) [docs] Split the Qwen3.8-27B NVFP4 cells by lm_head precision
- [#36203](https://github.com/sgl-project/sglang/pull/36203) Cleanup duplicate mamba backup helper
- [#35840](https://github.com/sgl-project/sglang/pull/35840) Add PD test for inkling with mxfp8 KV
- [#35957](https://github.com/sgl-project/sglang/pull/35957) Fix recurrent state loss on decode retraction
- [#25975](https://github.com/sgl-project/sglang/pull/25975) Fix prefill delayer wait histograms always observing 0
- [#36055](https://github.com/sgl-project/sglang/pull/36055) [Diffusion] Load MiniMax H3 GGUF text encoders
- [#36044](https://github.com/sgl-project/sglang/pull/36044) [Diffusion] Load Comfy NVFP4 MiniMax H3 checkpoints
- [#36040](https://github.com/sgl-project/sglang/pull/36040) [diffusion] feat: support mixed w4a4 and int8 checkpoints
- [#35929](https://github.com/sgl-project/sglang/pull/35929) Report the whole server's world size in the scheduler's internal state
- [#35928](https://github.com/sgl-project/sglang/pull/35928) Expose the declared sglang env vars of a scheduler in its internal state
- [#35927](https://github.com/sgl-project/sglang/pull/35927) Support gated launch to defer startup memory allocation
- [#35926](https://github.com/sgl-project/sglang/pull/35926) Report per-token weight-version spans in generation meta info
- [#35925](https://github.com/sgl-project/sglang/pull/35925) Make the scheduler track the published weight version
- [#35924](https://github.com/sgl-project/sglang/pull/35924) Extract _make_abort_req from the scheduler's abort paths
- [#35923](https://github.com/sgl-project/sglang/pull/35923) Extract collect_inflight_reqs from abort_request for reusing
- [#36056](https://github.com/sgl-project/sglang/pull/36056) [Diffusion] Load serialized FP8 CLIP image encoders
- [#36175](https://github.com/sgl-project/sglang/pull/36175) [diffusion] Fix test_model_fast_paths import after sana_ln_modulate rename
- [#36039](https://github.com/sgl-project/sglang/pull/36039) [Diffusion] Load serialized ConvRot W4A4 checkpoints
- [#35969](https://github.com/sgl-project/sglang/pull/35969) [diffusion] Accelerate LingBot Video RMSNorm in quality=high
- [#36171](https://github.com/sgl-project/sglang/pull/36171) [AMD][CI] Temporarily bypass local-registry image pulls

---
*Auto-generated by npu_pr_monitor.py*