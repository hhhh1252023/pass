# NPU PR 监控报告 (已合入)
**生成时间**: 2026-10-11 10:22 UTC
**本次检查已合入 PR 数**: 44
**涉及 NPU**: 12 | **无关**: 1 | **不确定**: 31

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#43346](https://github.com/sgl-project/sglang/pull/43346) [Sampler] Sort only the max(top_k) leading columns in the torch sampler
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#42857](https://github.com/sgl-project/sglang/pull/42857) [KDA] Support Helion MTP verification and harden prefill, decode
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#43228](https://github.com/sgl-project/sglang/pull/43228) [1/N][DSv41] Enable DP attention with MegaMoE
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#43633](https://github.com/sgl-project/sglang/pull/43633) [diffusion] fix: replay Qwen-Image-2.1 BCG graphs across prompt lengths
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#43586](https://github.com/sgl-project/sglang/pull/43586) [diffusion] torch.compile: keep Qwen-Image-2.1 and Ulysses graphs whole
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#43562](https://github.com/sgl-project/sglang/pull/43562) [PD][HiCache] Fold the decode restore outcome into the MIN-reduced KV poll
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#37907](https://github.com/sgl-project/sglang/pull/37907) [diffusion] Support LLaDA-Image BF16 generation and editing
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 28

### [#42957](https://github.com/sgl-project/sglang/pull/42957) [DSA] Support TileLang FP8 KV cache on CUDA
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#43564](https://github.com/sgl-project/sglang/pull/43564) [Fix] Reserve remaining tokens for ignore_eos requests during prefill admission
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#43495](https://github.com/sgl-project/sglang/pull/43495) [diffusion] comfyui: fail early instead of silently dropping features in SGLD integrated mode
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 13

### [#43602](https://github.com/sgl-project/sglang/pull/43602) [diffusion] torch.compile: keep the TP all-reduce inside the graph
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#43568](https://github.com/sgl-project/sglang/pull/43568) [diffusion] BCG: warn instead of disabling on unvalidated models
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

## ❓ 不确定是否涉及 NPU 的 PR

### [#43283](https://github.com/sgl-project/sglang/pull/43283) [Scheduler] Keep a DP-converted decode batch as the running batch
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43290](https://github.com/sgl-project/sglang/pull/43290) [Scheduler] Hold back decode rows on a mixed step when they do not fit the KV pool
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43286](https://github.com/sgl-project/sglang/pull/43286) [KV canary] Expect sweep counters on the target runner only
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43285](https://github.com/sgl-project/sglang/pull/43285) [Test] Do not hang scripted-runtime teardown when the server died
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43237](https://github.com/sgl-project/sglang/pull/43237) [Perf] Fuse BF16 writes to native HND KV cache storage
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43298](https://github.com/sgl-project/sglang/pull/43298) [Scheduler] Requeue preempted requests when admission admits nothing
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41933](https://github.com/sgl-project/sglang/pull/41933) [qwen 3.8 next] Fuse QSA prefill KV gathering and dtype conversion
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43037](https://github.com/sgl-project/sglang/pull/43037) [Spec] Allow Qwen3.5 backends to opt into packed target verification
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43479](https://github.com/sgl-project/sglang/pull/43479) [Scheduler] Keep waiting-prefix LRU refresh consistent across TP ranks
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43383](https://github.com/sgl-project/sglang/pull/43383) [AMD]Qwen add Sparse attention to capturing hip graphs
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43332](https://github.com/sgl-project/sglang/pull/43332) [GDN] Fix CuTe DSL decode to read the [N, HV, V, K] state layout
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43084](https://github.com/sgl-project/sglang/pull/43084) [diffusion] Don't report video jobs completed after their outputs are deleted
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43492](https://github.com/sgl-project/sglang/pull/43492) [diffusion] comfyui: consolidate executor conditioning-cache fixes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43490](https://github.com/sgl-project/sglang/pull/43490) [diffusion] comfyui: consolidate FluxAdapter correctness fixes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43634](https://github.com/sgl-project/sglang/pull/43634) [diffusion] MiniMax-H3: run the attention section eager under torch.compile
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43632](https://github.com/sgl-project/sglang/pull/43632) [diffusion] fix: Qwen-Image BCG prompt padding misses every graph and attends the pad
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43577](https://github.com/sgl-project/sglang/pull/43577) [diffusion] torch.compile: keep Cosmos3's fused kernels in the graph
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43631](https://github.com/sgl-project/sglang/pull/43631) [diffusion] torch.compile: serve whole graphs after offload-during-compile
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43590](https://github.com/sgl-project/sglang/pull/43590) [diffusion] Keep the overlay probe from stalling cached startup
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38217](https://github.com/sgl-project/sglang/pull/38217) fix: wait for HiCache load-back before hybrid MLA KV reads
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43601](https://github.com/sgl-project/sglang/pull/43601) [diffusion] torch.compile: skip inductor's saved multi-kernel pick (Z-Image warm starts crash or hang)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43574](https://github.com/sgl-project/sglang/pull/43574) test(diffusion): skip the LTX-2 RMSNorm-modulation guard test off the CUDA platform
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43501](https://github.com/sgl-project/sglang/pull/43501) [diffusion] Add the dynamic LoRA delta inside the second GEMM
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43576](https://github.com/sgl-project/sglang/pull/43576) [diffusion] Qwen-Image-2.1: skip QKV packing while layers are offloaded
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43584](https://github.com/sgl-project/sglang/pull/43584) [diffusion] IPC all-to-all: keep exported buffers out of expandable segments
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43565](https://github.com/sgl-project/sglang/pull/43565) [diffusion] Bind each GPU worker to its GPU's NUMA node
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43575](https://github.com/sgl-project/sglang/pull/43575) [diffusion] FLUX: declare regional compile regions
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43573](https://github.com/sgl-project/sglang/pull/43573) [diffusion] Accept request-level profiling on the image API
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42808](https://github.com/sgl-project/sglang/pull/42808) [Dsv4.1] Mega-mhc integration
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43624](https://github.com/sgl-project/sglang/pull/43624) [Model] Keep gpt-oss router logits batch-invariant under deterministic inference
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41591](https://github.com/sgl-project/sglang/pull/41591) [DCP + L3 3/N] Support hybrid MLA/KDA DCP with file and Mooncake HiCache L3
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#43598](https://github.com/sgl-project/sglang/pull/43598) [CI] Fix Kimi-Linear DCP PD test startup

---
*Auto-generated by npu_pr_monitor.py*