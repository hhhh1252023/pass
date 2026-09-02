# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-02 09:01 UTC
**本次检查已合入 PR 数**: 38
**涉及 NPU**: 8 | **无关**: 30 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#37504](https://github.com/sgl-project/sglang/pull/37504) [CI] Install sgl-eval from PyPI through the test extra
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 32

### [#37297](https://github.com/sgl-project/sglang/pull/37297) [Bugfix] Avoid scanning crash-dump token buffers during GC
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#34198](https://github.com/sgl-project/sglang/pull/34198) [AMD] Perf Kimi-K3 fuse ROCm KDA decode boundary
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#37258](https://github.com/sgl-project/sglang/pull/37258) Build Rust extensions on hosted runners in parallel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#37399](https://github.com/sgl-project/sglang/pull/37399) [NPU] Update sgl-kernel-npu version to 2026.9.0 and move memfabric deps into pyproject
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#37096](https://github.com/sgl-project/sglang/pull/37096) [Diffusion] Fuse FLUX.2 NVFP4 FC1, SwiGLU, and FC2 quantization
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#36865](https://github.com/sgl-project/sglang/pull/36865) [Kernel] Add KDA NVFP4 GEMM for Qwen3.x on SM120
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 24

### [#37331](https://github.com/sgl-project/sglang/pull/37331) Fix GPU kernel ordering and MXFP8 quantization dispatch
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

## ✅ 与 NPU 无关的已合入 PR
- [#37566](https://github.com/sgl-project/sglang/pull/37566) fix: restore missing get_component_forced_attn_backend import in minimax_h3
- [#34893](https://github.com/sgl-project/sglang/pull/34893) [Diffusion] Add MiniMax H3 cube sparse attention
- [#37441](https://github.com/sgl-project/sglang/pull/37441) [Diffusion] Admit explicit attention backends by capability
- [#37343](https://github.com/sgl-project/sglang/pull/37343) Fix nondeterministic FlashInfer GDN alignment test
- [#37327](https://github.com/sgl-project/sglang/pull/37327) Rust server: align launcher and request validation behavior
- [#37274](https://github.com/sgl-project/sglang/pull/37274) Allow custom policy for adaptive speculative decoding
- [#37340](https://github.com/sgl-project/sglang/pull/37340) [XPU] Add Regular Docker Image Release workflow for Intel XPU
- [#36646](https://github.com/sgl-project/sglang/pull/36646) [misc] Resolve SWA ownership at enqueue time for grouped free()
- [#37481](https://github.com/sgl-project/sglang/pull/37481) [mem_cache] Split duplicate insert frees at the SWA eviction floor
- [#37494](https://github.com/sgl-project/sglang/pull/37494) [Bugfix] Skip absent radix lock during cache cleanup
- [#37477](https://github.com/sgl-project/sglang/pull/37477) [Kernel] GLM 5.3 Flash related kernels (ported from #36507)
- [#37529](https://github.com/sgl-project/sglang/pull/37529) update CODEOWNERS
- [#37209](https://github.com/sgl-project/sglang/pull/37209) Add polisettyvarma into CI_PERMISSION list
- [#37335](https://github.com/sgl-project/sglang/pull/37335) [Fix ] Fix Spark2.5 hybrid SWA config
- [#35443](https://github.com/sgl-project/sglang/pull/35443) Fix reasoning metrics and add TPOT to bench_multiturn
- [#37518](https://github.com/sgl-project/sglang/pull/37518) [AMD][CI] Exclude unavailable MI355X nodes and skip 4N nightly
- [#37508](https://github.com/sgl-project/sglang/pull/37508) [CI] Preserve NCCL 2.30.7 after dependency installs
- [#37252](https://github.com/sgl-project/sglang/pull/37252) [CI] Batch CPU test workers
- [#37422](https://github.com/sgl-project/sglang/pull/37422) [Diffusion] Add cumulative extra-high quality tier
- [#32733](https://github.com/sgl-project/sglang/pull/32733) [CPU] Support FP8 KV cache
- [#37230](https://github.com/sgl-project/sglang/pull/37230) [XPU][CI] Enable nightly-xpu-8-gpu suite: declare + wire runner job
- [#36824](https://github.com/sgl-project/sglang/pull/36824) [Diffusion] Remove component loader capability switches
- [#37300](https://github.com/sgl-project/sglang/pull/37300) Decouple ragged CUDA graph request and token capacities
- [#37484](https://github.com/sgl-project/sglang/pull/37484) Temporarily Remove GLM-5.3 Flash prefill CP support
- [#37437](https://github.com/sgl-project/sglang/pull/37437) [Diffusion] Add SpargeAttention backend
- [#37302](https://github.com/sgl-project/sglang/pull/37302) [PD] Diversify fake-prefill handoff tokens
- [#37452](https://github.com/sgl-project/sglang/pull/37452) [CI] Double the base-b-test-1-gpu-large timeout
- [#37492](https://github.com/sgl-project/sglang/pull/37492) [Cookbook] Verify DeepSeek-V4 Flash Vision on GB300
- [#37486](https://github.com/sgl-project/sglang/pull/37486) test: add GLM-5.3 Flash DFlash2 B200 coverage
- [#36811](https://github.com/sgl-project/sglang/pull/36811) [Kernel] Avoid zero-bias allocation in fused softmax routing

---
*Auto-generated by npu_pr_monitor.py*