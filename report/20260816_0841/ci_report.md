# NPU CI 执行监控
**生成时间**: 2026-08-16 00:41 UTC
**分析 Run 数**: 63

---

## 📊 本次执行总结

- **成功 Job 数**: 417
- **失败 Run 数**: 44
- **成功 Job 平均耗时**: 29.1min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 272.3min | #31894414369 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95044204928) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 266.2min | #31882423785 | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95009406531) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 265.8min | #31873437889 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94988009012) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 263.1min | #31885957605 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95017880865) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 128.0min | #31871467376 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024960) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 124.7min | #31879765260 | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889560) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 124.2min | #31877531317 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433801) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 122.4min | #31906402708 | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878545) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 122.0min | #31889377090 | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304440) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 117.2min | #31894176235 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825820) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31894414369<br>[#34962 [Quantization] Fix GPTQ scheme attachment broken by LinearBase.scheme default](https://github.com/sgl-project/sglang/pull/34962) | `mmangkad/fix-gptq-scheme-attach` | 352.3min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31894414369) |
| #31873437889<br>[#34921 Suppress expected FlashInfer TRT-LLM workspace warnings](https://github.com/sgl-project/sglang/pull/34921) | `mmangkad/fix-flashinfer-allreduce-workspace-warning` | 300.0min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31873437889) |
| #31882423785<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `codex/k3-mm-cache-k3` | 294.4min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31882423785) |
| #31877531317<br>[#34916 [misc] Rename the WAR read-done fastpath to shared-read-done](https://github.com/sgl-project/sglang/pull/34916) | `lsyin/refactor-war-read-done` | 220.1min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31877531317) |
| #31894176235<br>[#34763 [Spec] Support mamba-radix-cache-strategy extra_buffer_lazy with DFLASH](https://github.com/sgl-project/sglang/pull/34763) | `dflash-extra-buffer-lazy` | 217.6min | 3 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31894176235) |
| #31879765260<br>[#34939 feat: support return_logprob under DSPARK speculative decoding](https://github.com/sgl-project/sglang/pull/34939) | `feature/dspark_logprob` | 192.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31879765260) |
| #31886913272<br>[#30531 [DSA] Skip indexer KV cache for skip-topk layers](https://github.com/sgl-project/sglang/pull/30531) | `mmangkad/reland-30310` | 191.1min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31886913272) |
| #31898996221<br>[#34870 Fix swa eviction frontier for bigram keys](https://github.com/sgl-project/sglang/pull/34870) | `fix-swa-tombstone-match` | 185.2min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31898996221) |
| #31889377090<br>[#33040 [minimax m3][npu]Adaptation of Minimax M3(w8a8) for NPU platforms [2/2]](https://github.com/sgl-project/sglang/pull/33040) | `main_fuseep` | 166.4min | 5 | base-b-test-16-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31889377090) |
| #31882695806<br>[#31575 Fix rope config compatibility and VL/transformers-fallback weight loading](https://github.com/sgl-project/sglang/pull/31575) | `fix/rope-config-and-vl-weight-loading` | 138.4min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31882695806) |
| #31906402708<br>[#30805 [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103](https://github.com/sgl-project/sglang/pull/30805) | `dsv4_fp8_trtllm_gen` | 128.3min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31906402708) |
| #31893841754<br>[#34668 fix(test): stabilize nightly precision regression](https://github.com/sgl-project/sglang/pull/34668) | `xinyuan/nightly-precision-stale-baseline` | 127.8min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31893841754) |
| #31896285786<br>[#34558 [Bugfix] Preserve MXFP4 Triton weights in sharded state](https://github.com/sgl-project/sglang/pull/34558) | `fix-mxfp4-sharded-state` | 125.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31896285786) |
| #31893629626<br>[#33726 fix(bcg): preserve Qwen3-VL DeepStack inputs during replay](https://github.com/sgl-project/sglang/pull/33726) | `fix/bcg-deepstack-replay-slot` | 118.2min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31893629626) |
| #31876732269<br>[#33676 [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management](https://github.com/sgl-project/sglang/pull/33676) | `main_8.5` | 115.3min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31876732269) |
| #31871829383<br>[#33576  [AMD] Add Work-Centric (Lean) Attention: a persistent-CTA decode kernel for long-context serving](https://github.com/sgl-project/sglang/pull/33576) | `wca-rebased` | 115.1min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31871829383) |
| #31874417154<br>[#34837 [AMD] Add concat_and_cast_mha_k_pad_kernel to support 12-head and enable K3 aiter prefill kernel](https://github.com/sgl-project/sglang/pull/34837) | `k3-aiter-prefill-kernel` | 105.3min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31874417154) |
| #31884125780<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 100.2min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31884125780) |
| #31873761013<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `codex/k3-mm-cache-k3` | 100.2min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31873761013) |
| #31877410827<br>[#34580 [AMD] Optimize KIMI-K3 with Triton MLA decode kernel by tuning the stage-1 geometry for gfx950](https://github.com/sgl-project/sglang/pull/34580) | `amd-mla-decode-gfx950-tune` | 99.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31877410827) |
| #31878529966<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `codex/k3-mm-cache-k3` | 93.1min | 3 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31878529966) |
| #31877721683<br>[#34926 Clean deprecated DeepSeek V4 Environs](https://github.com/sgl-project/sglang/pull/34926) | `clean-dsv4` | 91.8min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31877721683) |
| #31875731871<br>[#33685 [NPU CI] Reorganize test output/log directory structure with workflow context](https://github.com/sgl-project/sglang/pull/33685) | `pllimax/output-log-dir-structure` | 89.4min | 2 | base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31875731871) |
| #31874378798<br>[#34509 [JIT Kernel] Migrate moe_topk_softmax from AOT to JIT](https://github.com/sgl-project/sglang/pull/34509) | `voidc-minor/jit-moe-topk-softmax` | 87.8min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31874378798) |
| #31911064761<br>[#34916 [misc] Rename the WAR read-done fastpath to shared-read-done](https://github.com/sgl-project/sglang/pull/34916) | `main` | 64.0min | 5 | base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31911064761) |
| #31896606032<br>[#34870 Fix swa eviction frontier for bigram keys](https://github.com/sgl-project/sglang/pull/34870) | `fix-swa-tombstone-match` | 51.2min | 8 | base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31896606032) |
| #31894264221<br>[#34870 Fix swa eviction frontier for bigram keys](https://github.com/sgl-project/sglang/pull/34870) | `fix-swa-tombstone-match` | 51.0min | 6 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31894264221) |
| #31893883095<br>[#33604 Fix Whisper transcription for audio over 30 seconds](https://github.com/sgl-project/sglang/pull/33604) | `main` | 46.6min | 7 | base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31893883095) |
| #31903260080<br>[#33778 Avoid materializing GDN QKV tensors during target verification](https://github.com/sgl-project/sglang/pull/33778) | `perf/gdn-strided-target-verify` | 31.6min | 7 | base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31903260080) |
| #31913842643<br>[#30024 [AMD] perf(sgl-kernel): default block_quota=16 for MLA page_first KV gather…](https://github.com/sgl-project/sglang/pull/30024) | `main` | 30.0min | 8 | base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31913842643) |
| #31873565285<br>[#33685 [NPU CI] Reorganize test output/log directory structure with workflow context](https://github.com/sgl-project/sglang/pull/33685) | `pllimax/output-log-dir-structure` | 28.9min | 8 | base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31873565285) |
| #31875126739<br>[#34883 [Kimi-K3] Use explicit SiTU activation for MegaMoE](https://github.com/sgl-project/sglang/pull/34883) | `main` | 26.9min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31875126739) |
| #31898707796<br>[#34796 Add --http2-max-concurrent-streams server arg](https://github.com/sgl-project/sglang/pull/34796) | `main` | 23.2min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31898707796) |
| #31910041003<br>[#34837 [AMD] Add concat_and_cast_mha_k_pad_kernel to support 12-head and enable K3 aiter prefill kernel](https://github.com/sgl-project/sglang/pull/34837) | `main` | 23.0min | 9 | base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31910041003) |
| #31885550392<br>[#30531 [DSA] Skip indexer KV cache for skip-topk layers](https://github.com/sgl-project/sglang/pull/30531) | `mmangkad/reland-30310` | 22.2min | 9 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-a-test-1-npu-a2 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31885550392) |
| #31871115743<br>[#34270 [Review vehicle] config: the runner-side instance reads finish converging (not for merge)](https://github.com/sgl-project/sglang/pull/34270) | `cheng/gc-sr-review` | 20.9min | 9 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31871115743) |
| #31871506669<br>[#34517 [AMD][Spec] Accelerate Qwen3.5 verification with grouped-head shared KV](https://github.com/sgl-project/sglang/pull/34517) | `main` | 20.6min | 10 | base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31871506669) |
| #31876866287<br>[#34916 [misc] Rename the WAR read-done fastpath to shared-read-done](https://github.com/sgl-project/sglang/pull/34916) | `lsyin/refactor-war-read-done` | 16.1min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31876866287) |
| #31886502531<br>[#30531 [DSA] Skip indexer KV cache for skip-topk layers](https://github.com/sgl-project/sglang/pull/30531) | `mmangkad/reland-30310` | 9.5min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31886502531) |
| #31871531794<br>[#33676 [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management](https://github.com/sgl-project/sglang/pull/33676) | `main_8.5` | 8.5min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31871531794) |
| #31878253658<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `codex/k3-mm-cache-k3` | 6.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31878253658) |
| #31872408774<br>[#34263 config: the last runner-side instance reads read the bags](https://github.com/sgl-project/sglang/pull/34263) | `main` | 6.0min | 1 | base-a-test-1-npu-a2 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31872408774) |
| #31914836737<br>[#34982 [Fix] Delegate shared-read-ends declarations in wrapper attention backends](https://github.com/sgl-project/sglang/pull/34982) | `lsyin/shared-read-default-pre-replay` | 5.3min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31914836737) |
| #31871114383<br>[#34270 [Review vehicle] config: the runner-side instance reads finish converging (not for merge)](https://github.com/sgl-project/sglang/pull/34270) | `cheng/gc-sr-5-debt-ratchet` | 5.2min | 1 | check-changes | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31871114383) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 45 | 40 | 0 | 5 |
| base-b-test-1-npu-a3 / run (0) | 44 | 30 | 0 | 14 |
| base-b-test-16-npu-a3 / run (0) | 44 | 25 | 0 | 19 |
| base-b-test-2-npu-a3 / run (0) | 44 | 32 | 0 | 12 |
| base-b-test-4-npu-a3 / run (0) | 44 | 27 | 0 | 17 |
| base-b-test-4-npu-a3 / run (1) | 44 | 33 | 0 | 11 |
| base-b-test-8-npu-a3 / run (0) | 44 | 41 | 0 | 3 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 44 | 30 | 0 | 14 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 44 | 25 | 0 | 19 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44 | 28 | 0 | 16 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 44 | 40 | 0 | 4 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 30 | 5 | 0 | 25 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 25 | 7 | 0 | 18 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 28 | 11 | 0 | 17 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 40 | 3 | 0 | 37 |
| check-changes | 1 | 0 | 0 | 1 |
| multimodal-gen-test-1-npu-a3 | 53 | 40 | 2 | 11 |

---


## [Run #31914836737](https://github.com/sgl-project/sglang/actions/runs/31914836737)
- **分支**: `lsyin/shared-read-default-pre-replay`
- **总耗时**: 5.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31914836737

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089858) |
| base-b-test-8-npu-a3 / run (0) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089874) |
| base-b-test-2-npu-a3 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089885) |
| base-b-test-4-npu-a3 / run (0) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089898) |
| base-b-test-4-npu-a3 / run (1) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089944) |
| base-a-test-1-npu-a2 / run (0) | 4.4min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089945) |
| base-b-test-1-npu-a3 / run (0) | 4.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089994) |
| base-b-test-16-npu-a3 / run (0) | 4.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090073) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090085) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090094) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.4min | 环境问题 | 自定义容器启动失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090112) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090155) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因（如测试断言失败、环境异常等）被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089858

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089874

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重时（约12%进度）自定义容器实现执行失败，错误信息提示联系自托管runner管理员，属于NPU容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089885

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089898

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089944

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行测试前执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089945

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员。日志显示NPU内存正常（60.81GB），但容器在加载分片时崩溃，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085089994

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动自定义容器时失败，错误信息显示执行自定义容器实现失败，可能是NPU驱动、容器镜像或K8s资源调度问题，导致作业无法正常运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090073

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，导致作业启动失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090085

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090094

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装依赖后执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090112

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程文件（如模型权重、测试数据或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914836737/job/95085090155


## [Run #31913842643](https://github.com/sgl-project/sglang/actions/runs/31913842643)
- **分支**: `main`
- **总耗时**: 30.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31913842643

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 26.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817128) |
| base-b-test-4-npu-a3 / run (0) | 8.0min | 代码错误 | HiCache MLA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817149) |
| base-b-test-16-npu-a3 / run (0) | 24.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817197) |
| base-b-test-2-npu-a3 / run (0) | 25.9min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817232) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 25.0min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817420) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 27.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817494) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95083907300) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 0.7min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95085597181) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常（HTTP 200），但中途出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817128

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试在NPU上运行约281秒后失败，退出码为1，导致整个作业失败。具体失败原因需查看该测试文件的详细输出，可能是功能实现或测试断言问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817149

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试运行中容器突然报错"Executing the custom container implementation failed"，随后进入清理流程，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817197

- **base-b-test-2-npu-a3 / run (0)**: 日志显示torch_npu的transfer_to_npu模块在容器启动时抛出ImportWarning和RuntimeWarning，随后自定义容器实现执行失败，导致作业无法正常运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817232

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在Decode阶段后出现错误："Executing the custom container implementation failed. Please contact your self hosted runner administrator."，表明runner环境或容器配置存在问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817420

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但执行到23:35:24时出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817494

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 该作业在启动时被健康检查拦截，因同一运行中base-b-test-4-npu-a3作业失败而被判定为根因失败，触发fast-fail机制跳过执行，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95083907300

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示执行自定义容器实现时失败，提示联系自托管runner管理员，属于基础设施或容器环境配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95085597181

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 28.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817094) |
| base-b-test-8-npu-a3 / run (0) | 11.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817194) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817263) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817447) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31913842643/job/95082817455) |


## [Run #31911064761](https://github.com/sgl-project/sglang/actions/runs/31911064761)
- **分支**: `main`
- **总耗时**: 64.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31911064761

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 8.1min | 代码错误 | HiCache MLA测试文件执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132601) |
| base-b-test-16-npu-a3 / run (0) | 58.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132721) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 59.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132863) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 24.7min | 环境问题 | 自定义容器执行失败，NPU性能测试在启动阶段崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95080205505) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 16.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95081006457) |

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试在281秒后失败，0/5测试通过，具体错误信息未在日志中显示，但可确定是该测试文件本身存在问题导致执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132601

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载至37%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132721

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在23:05:44出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132863

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示bench_serving命令启动后，在请求模型信息时容器执行失败（Executing the custom container implementation failed），可能是NPU环境或容器配置问题，导致性能测试无法正常进行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95080205505

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示服务正常运行，但runner在23:05:44报错'Executing the custom container implementation failed'，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95081006457

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132579) |
| multimodal-gen-test-1-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132580) |
| base-b-test-2-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132623) |
| base-b-test-8-npu-a3 / run (0) | 11.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132634) |
| base-b-test-1-npu-a3 / run (0) | 46.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132663) |
| base-b-test-4-npu-a3 / run (1) | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132665) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132820) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132844) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076132855) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31911064761/job/95076941543) |


## [Run #31910041003](https://github.com/sgl-project/sglang/actions/runs/31910041003)
- **分支**: `main`
- **总耗时**: 23.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31910041003

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 8.3min | 代码错误 | HiCache MLA测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650254) |
| base-b-test-2-npu-a3 / run (0) | 22.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650267) |
| base-b-test-4-npu-a3 / run (1) | 21.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650279) |
| base-b-test-1-npu-a3 / run (0) | 21.6min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650301) |
| base-b-test-16-npu-a3 / run (0) | 21.8min | 环境问题 | NPU容器执行失败，模型权重加载时发生内存错误导致进程崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650357) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650538) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 18.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650545) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650583) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 15.6min | 环境问题 | 自定义容器执行失败，NPU性能测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95074322204) |

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test_npu_hicache_mla.py执行失败（退出码1），耗时291秒，导致整个作业失败。具体失败原因需查看该测试的详细输出，可能是功能实现或测试用例本身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650254

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试运行正常，但在22:02:08出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650267

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行正常（进度40%），但突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650279

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650301

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载MoE模型权重时，libtorch_cpu.so中发生内存访问错误，随后Scheduler watchdog超时，最终自定义容器执行失败。可能是NPU环境内存不足或驱动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650357

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650538

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，提示联系runner管理员，属于runner或容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650545

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在22:02:08出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650583

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示性能测试运行正常，但在22:02:08时自定义容器实现执行失败，提示联系自托管runner管理员，可能是容器环境或资源问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95074322204

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 11.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650252) |
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650296) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910041003/job/95073650532) |


## [Run #31906402708](https://github.com/sgl-project/sglang/actions/runs/31906402708)
- **分支**: `dsv4_fp8_trtllm_gen`
- **总耗时**: 128.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31906402708

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95065504552) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因同PR中另一作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95068215730) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95069515327) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95078787764) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1080秒后失败，0/1通过，属于性能测试未达标，可能因推理延迟或吞吐量不满足50ms要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95065504552

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（16-npu）因快速失败机制被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95068215730

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: PR健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业，导致本作业被快速失败跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95069515327

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业未实际运行测试，而是在健康检查阶段因根因作业base-c-test-perf-8-npu-a3失败而触发fast-fail跳过，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95078787764

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878265) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878338) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878346) |
| base-b-test-2-npu-a3 / run (0) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878351) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878360) |
| base-b-test-4-npu-a3 / run (1) | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878370) |
| base-b-test-16-npu-a3 / run (0) | 54.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878373) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878469) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 122.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878545) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878568) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878578) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906402708/job/95064878636) |


## [Run #31906355679](https://github.com/sgl-project/sglang/actions/runs/31906355679)
- **分支**: `codex/hunyuan3d-paint-native`
- **总耗时**: 25.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31906355679

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31906355679/job/95064830641) |


## [Run #31903260080](https://github.com/sgl-project/sglang/actions/runs/31903260080)
- **分支**: `perf/gdn-strided-target-verify`
- **总耗时**: 31.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31903260080

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195119) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195148) |
| base-b-test-4-npu-a3 / run (0) | 25.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195208) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195326) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195372) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | PR健康检查失败，lint检查未通过导致作业快速失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195414) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057920507) |

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195119

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业在运行测试前即被终止，属于CI前置检查失败而非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195148

- **base-b-test-4-npu-a3 / run (0)**: 日志显示torch_npu在transfer_to_npu时出现ImportWarning和RuntimeWarning，随后自定义容器实现执行失败，提示联系runner管理员，属于NPU容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195208

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但突然报错“Executing the custom container implementation failed”，提示联系runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195326

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在19:43:42时出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195372

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查状态为failure，触发fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195414

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段执行PR健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业在运行测试前即被终止，并非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057920507

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195093) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195129) |
| base-b-test-2-npu-a3 / run (0) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195165) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195218) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195252) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31903260080/job/95057195360) |


## [Run #31901159319](https://github.com/sgl-project/sglang/actions/runs/31901159319)
- **分支**: `codex/hunyuan3d-native-image-encoders`
- **总耗时**: 32.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31901159319

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31901159319/job/95052078249) |


## [Run #31900486135](https://github.com/sgl-project/sglang/actions/runs/31900486135)
- **分支**: `codex/glm-image-native-ar`
- **总耗时**: 41.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31900486135

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31900486135/job/95050401769) |


## [Run #31898996221](https://github.com/sgl-project/sglang/actions/runs/31898996221)
- **分支**: `fix-swa-tombstone-match`
- **总耗时**: 185.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31898996221

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.0min | 性能回归 | NPU性能测试未达预期，测试用例失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95047243298) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 28.1min | 性能回归 | NPU性能测试未通过，qwen3_235b_w8a8_8p_in3k5_out1k5_50ms测试失败，4个测试全部未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95051399864) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1232秒后退出码1，0/1通过，可能因性能未达标或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95047243298

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试脚本test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行失败，退出码1，耗时1461秒，4个测试全部失败，可能因性能未达预期或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95051399864

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 47.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705338) |
| base-b-test-2-npu-a3 / run (0) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705342) |
| multimodal-gen-test-1-npu-a3 | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705350) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705385) |
| base-b-test-8-npu-a3 / run (0) | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705392) |
| base-b-test-4-npu-a3 / run (1) | 13.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705403) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705458) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705558) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705564) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705569) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705570) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95046705608) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95051023718) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 77.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898996221/job/95058895781) |


## [Run #31898707796](https://github.com/sgl-project/sglang/actions/runs/31898707796)
- **分支**: `main`
- **总耗时**: 23.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31898707796

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916600) |
| base-b-test-1-npu-a3 / run (0) | 22.4min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916698) |
| base-b-test-2-npu-a3 / run (0) | 22.3min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916700) |
| base-b-test-4-npu-a3 / run (0) | 8.7min | 代码错误 | HiCache MLA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916717) |
| base-b-test-4-npu-a3 / run (1) | 21.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916733) |
| base-b-test-16-npu-a3 / run (0) | 7.0min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916767) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 22.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916816) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 15.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916911) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916937) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.9min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95046413546) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行或失败的具体错误，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916600

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916698

- **base-b-test-2-npu-a3 / run (0)**: 测试运行到第4个用例时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916700

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试文件在NPU上运行失败，耗时302秒，测试摘要显示0/5通过，具体失败原因需查看该测试文件的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916717

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中（进度约31%）时，runner报错“Executing the custom container implementation failed”，随后进入清理流程，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916733

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在准备阶段执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，而非代码或测试本身的问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916767

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后报错"Executing the custom container implementation failed"，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916816

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916911

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，错误提示'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916937

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在安装triton-ascend依赖时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境配置或运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95046413546

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916745) |
| base-b-test-8-npu-a3 / run (0) | 11.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916746) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898707796/job/95045916908) |


## [Run #31898542760](https://github.com/sgl-project/sglang/actions/runs/31898542760)
- **分支**: `codex/hunyuan3d-native-image-encoders`
- **总耗时**: 30.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31898542760

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31898542760/job/95045577004) |


## [Run #31896606032](https://github.com/sgl-project/sglang/actions/runs/31896606032)
- **分支**: `fix-swa-tombstone-match`
- **总耗时**: 51.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31896606032

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 14.4min | 环境问题 | 自定义容器执行失败，NPU环境或容器配置异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836056) |
| base-b-test-2-npu-a3 / run (0) | 13.8min | 环境问题 | 自定义容器执行失败，NPU后端不支持CUDA设备类型导致服务异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836072) |
| base-b-test-16-npu-a3 / run (0) | 1.7min | 环境问题 | 自定义容器执行失败，可能是容器环境或资源问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836085) |
| base-b-test-4-npu-a3 / run (0) | 15.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836142) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 16.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836286) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836353) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.8min | 性能回归 | NPU性能测试未达预期，minimax_m2_5测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95041794129) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 8.4min | 环境问题 | 自定义容器执行失败，模型权重加载过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95044246667) |

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU设备、容器镜像或运行时环境问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836056

- **base-b-test-2-npu-a3 / run (0)**: 日志显示SymmetricMemory不支持cuda设备类型，且NPU后端对aten::_assert_async算子回退到CPU，最终自定义容器实现执行失败，属于环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836072

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在安装Rust组件后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836085

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在模型捕获批次过程中，自定义容器实现执行失败，错误信息为“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836142

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（decode吞吐正常），但突然报错“Executing the custom container implementation failed”，提示联系runner管理员，属于runner容器环境故障，非代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836286

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836353

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1139秒后失败，该测试为性能测试，失败原因可能是性能未达标或执行错误，需查看详细日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95041794129

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示在加载模型分片（约55%）时出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95044246667

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836007) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836086) |
| base-b-test-8-npu-a3 / run (0) | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836112) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836140) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836289) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896606032/job/95040836302) |


## [Run #31896285786](https://github.com/sgl-project/sglang/actions/runs/31896285786)
- **分支**: `fix-mxfp4-sharded-state`
- **总耗时**: 125.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31896285786

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95044915651) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现同批次其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95048524020) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 环境问题 | 上游作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95049097529) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95054315176) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1133秒后退出码1，0/1通过，属于性能指标未达标导致的回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95044915651

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，导致本作业（16-npu）被快速失败跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95048524020

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 本作业因健康检查检测到base-c-test-perf-8-npu-a3作业失败而被快速失败机制跳过，并非自身测试失败，属于级联取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95049097529

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的健康检查中检测到根因作业 base-c-test-perf-8-npu-a3 失败，因此被快速失败机制跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95054315176

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038446) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038529) |
| base-b-test-2-npu-a3 / run (0) | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038533) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038545) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038558) |
| base-b-test-16-npu-a3 / run (0) | 54.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038570) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038579) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038582) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038754) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038840) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040038859) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31896285786/job/95040039053) |


## [Run #31895887803](https://github.com/sgl-project/sglang/actions/runs/31895887803)
- **分支**: `codex/pi05-native-siglip`
- **总耗时**: 51.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31895887803

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31895887803/job/95039153874) |


## [Run #31894414369](https://github.com/sgl-project/sglang/actions/runs/31894414369)
- **分支**: `mmangkad/fix-gptq-scheme-attach`
- **总耗时**: 352.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31894414369

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未通过，minimax_m2_5模型w8a8配置下耗时超预期。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95041884859) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95048713187) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1192秒后失败，返回码1，0/1测试通过，属于性能指标未达标导致的回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95041884859

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到同一次运行中的base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，导致本作业在启动前被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95048713187

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637085) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637091) |
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637104) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637105) |
| base-b-test-16-npu-a3 / run (0) | 51.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637130) |
| base-b-test-4-npu-a3 / run (0) | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637137) |
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637145) |
| base-b-test-4-npu-a3 / run (1) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637195) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637274) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637376) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637404) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95035637459) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95043610758) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 272.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894414369/job/95044204928) |


## [Run #31894264221](https://github.com/sgl-project/sglang/actions/runs/31894264221)
- **分支**: `fix-swa-tombstone-match`
- **总耗时**: 51.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31894264221

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 17.5min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050133) |
| base-b-test-16-npu-a3 / run (0) | 50.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050278) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 50.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050551) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 31.7min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050577) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 34.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050622) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95039276420) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050133

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，可能是临时文件清理或配置问题，需检查作业依赖的工件或缓存是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050278

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050551

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（decode吞吐约435 token/s），但在16:50:07时容器执行报错，提示联系self-hosted runner管理员，属于运行环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050577

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在处理请求时出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050622

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95039276420

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050193) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050216) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050217) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050221) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050255) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050277) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894264221/job/95035050673) |


## [Run #31894176235](https://github.com/sgl-project/sglang/actions/runs/31894176235)
- **分支**: `dflash-extra-buffer-lazy`
- **总耗时**: 217.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31894176235

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 30.0min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825631) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.4min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95037758456) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 47.3min | 性能回归 | 性能测试中qwen3_235b用例失败，疑似性能不达标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95041611610) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、Node版本警告及上传diffusion-failures目录（无文件）的步骤，未展示任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825631

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1073秒后退出码1，属于性能测试未通过，可能因推理速度或延迟未达50ms目标导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95037758456

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示qwen3_235b_a22b测试用例退出码1，耗时1488秒，而其他两个用例通过。该用例为性能测试，失败可能因吞吐或延迟未达阈值，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95041611610

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 52.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825577) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825637) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825659) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825670) |
| base-b-test-4-npu-a3 / run (0) | 33.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825674) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825686) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825801) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 117.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825820) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825898) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825907) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 10.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95034825937) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95039395395) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31894176235/job/95048552542) |


## [Run #31893883095](https://github.com/sgl-project/sglang/actions/runs/31893883095)
- **分支**: `main`
- **总耗时**: 46.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31893883095

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 39.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131076) |
| base-b-test-16-npu-a3 / run (0) | 42.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131115) |
| base-b-test-4-npu-a3 / run (0) | 8.5min | 代码错误 | HiCache MLA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131214) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 40.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131337) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95035004308) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95038708332) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95039176440) |

- **base-b-test-1-npu-a3 / run (0)**: 测试服务启动并成功响应请求后，runner在执行自定义容器实现时失败，提示联系管理员，属于自托管runner环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131076

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131115

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试文件在NPU上运行失败，耗时281秒，测试总结显示0/5通过。具体失败原因需查看该测试文件的详细输出，可能是功能实现或测试断言问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131214

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，可能是容器环境或自托管runner问题，导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131337

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1099秒后退出码为1，0/1通过，属于性能指标未达预期，可能因模型推理速度或延迟不满足要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95035004308

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95038708332

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95039176440

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131050) |
| base-b-test-8-npu-a3 / run (0) | 11.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131108) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131119) |
| base-b-test-2-npu-a3 / run (0) | 26.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131142) |
| base-b-test-4-npu-a3 / run (1) | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131191) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131348) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131356) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 36.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893883095/job/95034131423) |


## [Run #31893841754](https://github.com/sgl-project/sglang/actions/runs/31893841754)
- **分支**: `xinyuan/nightly-precision-stale-baseline`
- **总耗时**: 127.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31893841754

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.2min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034859055) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因依赖的另一个作业失败而被快速跳过（fast-fail）。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95037139957) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95039703542) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95046811928) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1177秒后退出码为1，0/1测试通过，属于性能指标未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034859055

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业为 base-c-test-perf-8-npu-a3，本作业（16-npu）被快速失败机制跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95037139957

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动阶段被PR健康检查拦截，原因是同一次运行中base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95039703542

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业在健康检查阶段检测到根因作业base-c-test-perf-8-npu-a3失败，触发快速失败机制，本作业被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95046811928

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029634) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029663) |
| base-b-test-4-npu-a3 / run (0) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029671) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029691) |
| base-b-test-16-npu-a3 / run (0) | 53.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029701) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029717) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029719) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029888) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 108.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029891) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029908) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893841754/job/95034029965) |


## [Run #31893629626](https://github.com/sgl-project/sglang/actions/runs/31893629626)
- **分支**: `fix/bcg-deepstack-replay-slot`
- **总耗时**: 118.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31893629626

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033990754) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95036599829) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95038530180) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95044476345) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，耗时1062秒，未通过性能阈值（50ms），可能因模型推理延迟超标或环境性能波动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033990754

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示PR健康检查检测到另一个作业base-c-test-perf-8-npu-a3失败，被判定为根因作业，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95036599829

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动阶段因健康检查检测到其他根因作业（base-c-test-perf-8-npu-a3）失败而触发快速失败机制，自身未实际运行测试，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95038530180

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95044476345

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522047) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522051) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522064) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522070) |
| multimodal-gen-test-1-npu-a3 | 26.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522086) |
| base-b-test-4-npu-a3 / run (0) | 33.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522087) |
| base-b-test-16-npu-a3 / run (0) | 55.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522126) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522127) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522175) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522176) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 94.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522188) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893629626/job/95033522210) |


## [Run #31893100470](https://github.com/sgl-project/sglang/actions/runs/31893100470)
- **分支**: `codex/sana-wm-native-refiner`
- **总耗时**: 28.1min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31893100470

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31893100470/job/95032303710) |


## [Run #31890556848](https://github.com/sgl-project/sglang/actions/runs/31890556848)
- **分支**: `codex/minimax-h3-reference-rng`
- **总耗时**: 33.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31890556848

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31890556848/job/95026112749) |


## [Run #31889377090](https://github.com/sgl-project/sglang/actions/runs/31889377090)
- **分支**: `main_fuseep`
- **总耗时**: 166.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31889377090

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他作业根因失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304244) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8.7min | 性能回归 | NPU性能测试未达预期，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023818361) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 上游作业失败导致快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95025816830) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95027832923) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95037394496) |

- **base-b-test-16-npu-a3 / run (0)**: 健康检查显示base-c-test-perf-8-npu-a3作业为根因失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304244

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，测试摘要显示0/1通过，耗时315秒，表明性能指标未满足要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023818361

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 本作业未实际运行，因健康检查发现同批次base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制被跳过，属于依赖的上游失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95025816830

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在健康检查阶段因检测到base-c-test-perf-8-npu-a3作业失败而触发快速失败机制，本作业被跳过未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95027832923

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95037394496

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304189) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304210) |
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304213) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304254) |
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304278) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304282) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304307) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304409) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304435) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 122.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304440) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889377090/job/95023304487) |


## [Run #31889115628](https://github.com/sgl-project/sglang/actions/runs/31889115628)
- **分支**: `codex/minimax-h3-vae-usp`
- **总耗时**: 27.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31889115628

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31889115628/job/95022674896) |


## [Run #31888756841](https://github.com/sgl-project/sglang/actions/runs/31888756841)
- **分支**: `main`
- **总耗时**: 33.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31888756841

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31888756841/job/95021787164) |


## [Run #31886913272](https://github.com/sgl-project/sglang/actions/runs/31886913272)
- **分支**: `mmangkad/reland-30310`
- **总耗时**: 191.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31886913272

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.2min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95018011012) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 47.5min | 性能回归 | qwen3_235b_a22b性能测试未通过，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95020115277) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1064秒，未达到性能预期标准，导致整个性能测试套件0/1通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95018011012

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中qwen3_235b_a22b测试失败（exit code 1），而其他两个测试通过。该测试耗时1467秒，可能未达到性能目标（50ms），属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95020115277

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017536916) |
| base-a-test-1-npu-a2 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017536918) |
| base-b-test-16-npu-a3 / run (0) | 50.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017536924) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017536955) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017536967) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017536997) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017537027) |
| base-b-test-8-npu-a3 / run (0) | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017537064) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017537160) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017537176) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017537177) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95017537263) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95021618638) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 78.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886913272/job/95029582649) |


## [Run #31886502531](https://github.com/sgl-project/sglang/actions/runs/31886502531)
- **分支**: `mmangkad/reland-30310`
- **总耗时**: 9.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31886502531

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535778) |
| base-b-test-4-npu-a3 / run (1) | 7.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535817) |
| base-b-test-4-npu-a3 / run (0) | 7.7min | 环境问题 | 自定义容器启动失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535818) |
| base-b-test-16-npu-a3 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535835) |
| base-b-test-2-npu-a3 / run (0) | 6.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535842) |
| base-b-test-1-npu-a3 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535876) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.3min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016536089) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.5min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016536110) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016536238) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 3.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95017105552) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535778

- **base-b-test-4-npu-a3 / run (1)**: 日志显示Prefill batch正常执行，但随后出现'Executing the custom container implementation failed'错误，提示联系self-hosted runner管理员，属于NPU容器环境问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535817

- **base-b-test-4-npu-a3 / run (0)**: 日志显示torch_npu在transfer_to_npu时出现警告，随后自定义容器实现执行失败，提示联系runner管理员，属于NPU容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535818

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型权重加载过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535835

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在TokenizerManager初始化后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535842

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在分配KV缓存主机内存后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535876

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016536089

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型加载和初始化正常，但在TP1加载权重后，GitHub Actions报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016536110

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016536238

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95017105552

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535843) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016535853) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886502531/job/95016536090) |


## [Run #31886314390](https://github.com/sgl-project/sglang/actions/runs/31886314390)
- **分支**: `codex/ernie-pe-native`
- **总耗时**: 52.2min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31886314390

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31886314390/job/95016094765) |


## [Run #31885957605](https://github.com/sgl-project/sglang/actions/runs/31885957605)
- **分支**: `mhc`
- **总耗时**: 311.2min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31885957605

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212194) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212224) |
| base-b-test-16-npu-a3 / run (0) | 54.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212233) |
| base-b-test-2-npu-a3 / run (0) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212245) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212259) |
| base-b-test-1-npu-a3 / run (0) | 25.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212263) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212290) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212323) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212372) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212384) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212393) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015212411) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95015709931) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 263.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95017880865) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95019883497) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 74.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885957605/job/95024490312) |


## [Run #31885550392](https://github.com/sgl-project/sglang/actions/runs/31885550392)
- **分支**: `mmangkad/reland-30310`
- **总耗时**: 22.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31885550392

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 21.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245158) |
| base-b-test-4-npu-a3 / run (0) | 21.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245182) |
| base-b-test-1-npu-a3 / run (0) | 21.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245223) |
| base-b-test-16-npu-a3 / run (0) | 1.3min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245239) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245240) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 21.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245243) |
| base-a-test-1-npu-a2 / run (0) | 2.1min | 环境问题 | rustup 下载工具链超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245260) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 21.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245285) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014726368) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245158

- **base-b-test-4-npu-a3 / run (0)**: 测试运行到第3个用例时，自定义容器实现执行失败，导致作业提前终止。日志显示容器环境存在问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245182

- **base-b-test-1-npu-a3 / run (0)**: 日志显示torch_npu的transfer_to_npu模块在容器启动时产生ImportWarning和RuntimeWarning，随后自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245223

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业 base-a-test-1-npu-a2 / run (0)，因此本作业（base-b-test-16-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245239

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245240

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245243

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 1.92 时，从内部缓存服务下载 channel-rust-1.92.toml 超时，导致脚本退出码非零，作业失败。属于网络或缓存服务临时故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245260

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245285

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014726368

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245179) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245180) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245192) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31885550392/job/95014245258) |


## [Run #31884479074](https://github.com/sgl-project/sglang/actions/runs/31884479074)
- **分支**: `codex/minimax-h3-vae-usp`
- **总耗时**: 33.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31884479074

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884479074/job/95011958099) |


## [Run #31884125780](https://github.com/sgl-project/sglang/actions/runs/31884125780)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 100.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31884125780

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790287) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.6min | 精度回归 | Qwen3-VL-8B-Thinking MMMU精度测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790315) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 57.6min | 性能回归 | NPU性能测试中qwen3_235b_a22b用例失败，可能因性能未达预期或运行错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790321) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1121秒后失败，该测试为性能测试，预期时间3600秒，实际未通过，可能因性能指标未达要求或运行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790287

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 性能测试全部通过，但精度测试test_npu_qwen3_vl_8b_thinking_1p_mmmu.py退出码为1，耗时2438秒，导致整个作业失败，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790315

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中3个用例通过，但test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py退出码1，耗时1460秒，疑似性能不达标或执行异常，需检查该用例日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790321

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790086) |
| base-b-test-16-npu-a3 / run (0) | 48.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790113) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790126) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790153) |
| base-b-test-4-npu-a3 / run (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790164) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790204) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790214) |
| multimodal-gen-test-1-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790261) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790275) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790285) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790375) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790392) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31884125780/job/95023790426) |


## [Run #31883602297](https://github.com/sgl-project/sglang/actions/runs/31883602297)
- **分支**: `codex/qwen3vl-native-vision`
- **总耗时**: 30.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31883602297

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31883602297/job/95009596489) |


## [Run #31882695806](https://github.com/sgl-project/sglang/actions/runs/31882695806)
- **分支**: `fix/rope-config-and-vl-weight-loading`
- **总耗时**: 138.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31882695806

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95013365438) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因同PR中另一个作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95014602270) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95016127875) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95020812857) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1154秒后退出码1，0/1通过，疑似性能未达标或环境异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95013365438

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95014602270

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95016127875

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在健康检查阶段因根因作业base-c-test-perf-8-npu-a3失败而被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95020812857

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 11.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416774) |
| base-b-test-4-npu-a3 / run (0) | 26.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416782) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416800) |
| multimodal-gen-test-1-npu-a3 | 26.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416815) |
| base-b-test-2-npu-a3 / run (0) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416822) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416824) |
| base-b-test-16-npu-a3 / run (0) | 51.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416839) |
| base-a-test-1-npu-a2 / run (0) | 8.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416849) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416894) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416907) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416916) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882695806/job/95007416933) |


## [Run #31882423785](https://github.com/sgl-project/sglang/actions/runs/31882423785)
- **分支**: `codex/k3-mm-cache-k3`
- **总耗时**: 294.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31882423785

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95008992507) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95018783924) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1111秒后返回退出码1，0/1测试通过，属于性能测试未达标或执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95008992507

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到 `base-c-test-perf-8-npu-a3` 作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身执行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95018783924

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821544) |
| base-a-test-1-npu-a2 / run (0) | 12.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821591) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821613) |
| base-b-test-16-npu-a3 / run (0) | 49.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821620) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821629) |
| base-b-test-4-npu-a3 / run (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821631) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821649) |
| base-b-test-4-npu-a3 / run (0) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821670) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 115.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821799) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821807) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821856) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95006821878) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 266.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95009406531) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31882423785/job/95010876275) |


## [Run #31880165399](https://github.com/sgl-project/sglang/actions/runs/31880165399)
- **分支**: `codex/qwen25vl-native-generation`
- **总耗时**: 29.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31880165399

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31880165399/job/95001540834) |


## [Run #31879765260](https://github.com/sgl-project/sglang/actions/runs/31879765260)
- **分支**: `feature/dspark_logprob`
- **总耗时**: 192.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31879765260

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未达标，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95011932532) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现同PR中另一个作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95012823620) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95014603687) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败机制触发，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95025042709) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1138秒后退出码1，0/1通过，属于性能指标未达预期或执行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95011932532

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，将其视为根因作业，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95012823620

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95014603687

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业作为级联失败被快速失败机制跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95025042709

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889346) |
| base-b-test-16-npu-a3 / run (0) | 47.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889411) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889415) |
| base-b-test-1-npu-a3 / run (0) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889432) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889434) |
| base-a-test-1-npu-a2 / run (0) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889456) |
| base-b-test-4-npu-a3 / run (0) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889476) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889487) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 124.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889560) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889574) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889585) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31879765260/job/95006889647) |


## [Run #31878529966](https://github.com/sgl-project/sglang/actions/runs/31878529966)
- **分支**: `codex/k3-mm-cache-k3`
- **总耗时**: 93.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31878529966

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741953) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达预期，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94998894434) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 28.6min | 性能回归 | NPU性能测试未通过，测试用例执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/95001339517) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境问题而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741953

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1112秒后退出码为1，0/1通过，属于性能指标未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94998894434

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py返回退出码1，0/4测试通过，耗时1427秒，可能因性能未达预期或运行错误导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878529966/job/95001339517

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741718) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741764) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741800) |
| base-b-test-16-npu-a3 / run (0) | 54.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741820) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741829) |
| base-b-test-4-npu-a3 / run (1) | 13.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741830) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741837) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997741849) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997742001) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997742031) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/94997742035) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878529966/job/95001980135) |


## [Run #31878471462](https://github.com/sgl-project/sglang/actions/runs/31878471462)
- **分支**: `codex/fix-attention-backend-fallback`
- **总耗时**: 39.6min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31878471462

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878471462/job/94997562284) |


## [Run #31878441811](https://github.com/sgl-project/sglang/actions/runs/31878441811)
- **分支**: `codex/qwen25vl-native-generation`
- **总耗时**: 35.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31878441811

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878441811/job/94997500047) |


## [Run #31878253658](https://github.com/sgl-project/sglang/actions/runs/31878253658)
- **分支**: `codex/k3-mm-cache-k3`
- **总耗时**: 6.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31878253658

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064813) |
| base-b-test-4-npu-a3 / run (1) | 1.9min | 环境问题 | 自托管runner在安装依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064892) |
| base-b-test-4-npu-a3 / run (0) | 5.1min | 环境问题 | 自定义容器执行失败，NPU测试中途崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064902) |
| base-b-test-1-npu-a3 / run (0) | 6.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064910) |
| base-b-test-8-npu-a3 / run (0) | 6.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064936) |
| base-b-test-2-npu-a3 / run (0) | 6.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064947) |
| base-b-test-16-npu-a3 / run (0) | 6.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064981) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997065109) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997065116) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997065121) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997065141) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064813

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在pip安装过程中卸载旧包时，自定义容器实现执行失败，提示联系runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064892

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在捕获批次（bs=208降至160）过程中，可用内存从8.71GB缓慢下降至8.66GB，随后自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064902

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064910

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064936

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064947

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI依赖的某个工件或缓存文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，需检查相关依赖配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064981

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997065109

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储中找到，可能是资源被清理、路径错误或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997065116

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997065121

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997065141

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31878253658/job/94997064974) |


## [Run #31877721683](https://github.com/sgl-project/sglang/actions/runs/31877721683)
- **分支**: `clean-dsv4`
- **总耗时**: 91.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31877721683

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.9min | 性能回归 | NPU性能测试未达预期，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94996336254) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94998190568) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94999904687) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/95004353257) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py失败，耗时1051秒，未通过性能指标要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94996336254

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94998190568

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 本作业在启动前的PR健康检查中检测到同PR的base-c-test-perf-8-npu-a3作业已失败，被判定为级联失败，因此主动跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94999904687

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被过滤，最终因快速失败机制被跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877721683/job/95004353257

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995857839) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995857842) |
| multimodal-gen-test-1-npu-a3 | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995857869) |
| base-b-test-16-npu-a3 / run (0) | 57.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995857874) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995857893) |
| base-b-test-4-npu-a3 / run (0) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995857925) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995857927) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995858004) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995858024) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995858036) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995858059) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877721683/job/94995858068) |


## [Run #31877531317](https://github.com/sgl-project/sglang/actions/runs/31877531317)
- **分支**: `lsyin/refactor-war-read-done`
- **总耗时**: 220.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31877531317

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995997012) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 56.5min | 性能回归 | NPU性能测试中qwen3_235b_a22b用例失败，退出码1，其余用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94998793285) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1095秒后失败，0/1通过，属于性能测试未达标，可能因模型推理速度或吞吐量低于阈值导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995997012

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms性能测试未通过（exit code 1），其他三个性能用例均通过，判断为该模型性能未达标或执行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94998793285

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433641) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433647) |
| base-b-test-16-npu-a3 / run (0) | 55.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433653) |
| base-b-test-2-npu-a3 / run (0) | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433659) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433677) |
| multimodal-gen-test-1-npu-a3 | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433697) |
| base-b-test-4-npu-a3 / run (1) | 13.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433716) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433727) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433738) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433739) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433743) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 124.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94995433801) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/94999647060) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 75.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877531317/job/95009314290) |


## [Run #31877410827](https://github.com/sgl-project/sglang/actions/runs/31877410827)
- **分支**: `amd-mla-decode-gfx950-tune`
- **总耗时**: 99.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31877410827

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995961852) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现同PR中另一个作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94997549185) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94999013125) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/95003672136) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1133秒后退出码1，0/1通过，表明性能指标未达标，可能因模型推理速度或延迟超出预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995961852

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94997549185

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而触发fast-fail，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94999013125

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的PR健康检查中发现根因作业base-c-test-perf-8-npu-a3失败，触发fast-fail机制，本作业被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31877410827/job/95003672136

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112571) |
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112637) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112675) |
| base-b-test-8-npu-a3 / run (0) | 8.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112693) |
| base-b-test-4-npu-a3 / run (0) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112706) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112709) |
| base-b-test-16-npu-a3 / run (0) | 55.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112724) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112750) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112792) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112818) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112823) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31877410827/job/94995112827) |


## [Run #31876866287](https://github.com/sgl-project/sglang/actions/runs/31876866287)
- **分支**: `lsyin/refactor-war-read-done`
- **总耗时**: 16.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31876866287

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 13.3min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后容器异常退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836696) |
| base-b-test-1-npu-a3 / run (0) | 13.2min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836721) |
| multimodal-gen-test-1-npu-a3 | 14.3min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836726) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836759) |
| base-b-test-4-npu-a3 / run (0) | 14.1min | 环境问题 | 自托管runner容器执行失败，测试中途被中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836779) |
| base-b-test-16-npu-a3 / run (0) | 2.4min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836787) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836871) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.2min | 环境问题 | 自定义容器执行失败，NPU图捕获过程中容器崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836975) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993837160) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 9.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94994457196) |

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重（4/4 shards完成）后，出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836696

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836721

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时无文件，可能测试通过或未生成失败产物，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836726

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试在运行EAGLE3投机解码时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836759

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试在运行DPAttentionMixedChunk.test_gsm8k时，容器执行被中断，报错'Executing the custom container implementation failed'，属于runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836779

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在安装triton-ascend等依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836787

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，提示联系runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836871

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在NPU图捕获进行到17%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是容器资源限制或NPU驱动问题导致崩溃。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836975

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行中容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993837160

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94994457196

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836737) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836752) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876866287/job/94993836968) |


## [Run #31876749132](https://github.com/sgl-project/sglang/actions/runs/31876749132)
- **分支**: `main`
- **总耗时**: 34.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31876749132

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876749132/job/94993552602) |


## [Run #31876732269](https://github.com/sgl-project/sglang/actions/runs/31876732269)
- **分支**: `main_8.5`
- **总耗时**: 115.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31876732269

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.7min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993949128) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因同批次其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94995868102) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94997564050) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/95003559629) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1095秒后失败，该测试为性能测试，预期耗时3600秒，但未通过性能指标要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993949128

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（16-npu）被快速失败机制跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94995868102

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（4-npu）作为级联失败被快速跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94997564050

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的PR健康检查中发现根因作业base-c-test-perf-8-npu-a3失败，触发fast-fail机制，本作业被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31876732269/job/95003559629

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 37.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514636) |
| base-b-test-16-npu-a3 / run (0) | 49.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514649) |
| base-b-test-4-npu-a3 / run (0) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514651) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514661) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514674) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514686) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514716) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514728) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514753) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 101.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514841) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514878) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31876732269/job/94993514880) |


## [Run #31875731871](https://github.com/sgl-project/sglang/actions/runs/31875731871)
- **分支**: `pllimax/output-log-dir-structure`
- **总耗时**: 89.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31875731871

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 46.4min | 性能回归 | NPU性能测试中qwen3_235b用例失败，退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384878) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384894) |

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py测试失败（exit code 1），其他两个用例通过，可能因性能未达阈值或运行错误导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384878

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1141秒后失败，该测试为性能测试，失败原因可能是性能未达标或超时，需检查具体性能指标。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384894

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384540) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384559) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384677) |
| base-b-test-16-npu-a3 / run (0) | 52.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384700) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384703) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384752) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384772) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384829) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384830) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384882) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384922) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384933) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875731871/job/95020384959) |


## [Run #31875126739](https://github.com/sgl-project/sglang/actions/runs/31875126739)
- **分支**: `main`
- **总耗时**: 26.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31875126739

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 19.1min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634551) |
| base-b-test-16-npu-a3 / run (0) | 1.2min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634618) |
| base-b-test-4-npu-a3 / run (1) | 20.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634619) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 代码错误 | HiCache MLA 测试用例执行失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634660) |
| base-b-test-2-npu-a3 / run (0) | 20.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634665) |
| base-b-test-1-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634692) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 16.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634704) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634711) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634731) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94990524030) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未执行或全部通过，但作业被标记为失败，需查看更完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634551

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行PR测试健康检查时，检测到base-b-test-4-npu-a3作业失败（根因），本作业作为级联失败被跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634618

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中突然出现'Executing the custom container implementation failed'错误，提示联系runner管理员，属于自托管runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634619

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 在运行约292秒后失败，0/5测试通过，具体错误信息未在日志中显示，可能是功能实现或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634660

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试运行中容器突然报错“Executing the custom container implementation failed”，随后进入清理流程，属于runner或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634665

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到base-b-test-4-npu-a3作业失败，根因作业被过滤后触发fast-fail，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634692

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634704

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634711

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（吞吐量正常），但中途报错"Executing the custom container implementation failed"，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634731

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-4-npu-a3/run，触发fast-fail机制，本作业未实际运行即被终止，属于级联跳过而非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94990524030

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 11.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634585) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634724) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31875126739/job/94989634766) |


## [Run #31874670025](https://github.com/sgl-project/sglang/actions/runs/31874670025)
- **分支**: `agent/minimax-h3-b300-high-quality`
- **总耗时**: 31.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31874670025

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874670025/job/94988539945) |


## [Run #31874417154](https://github.com/sgl-project/sglang/actions/runs/31874417154)
- **分支**: `k3-aiter-prefill-kernel`
- **总耗时**: 105.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31874417154

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | 性能回归 | NPU性能测试未达预期，测试用例执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94989681716) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 27.7min | 性能回归 | NPU性能测试未达标，qwen3_235b_w8a8_8p测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94991116699) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94993537176) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94998041106) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py退出码为1，耗时1136秒，未通过性能基准，可能因模型性能下降或环境波动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94989681716

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行1394秒后失败，0/4测试通过，可能因性能未达预期或环境问题导致退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94991116699

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（base-c-test-perf-4-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94993537176

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现base-c-test-perf-8-npu-a3和base-c-test-perf-16-npu-a3两个根因作业失败，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94998041106

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 52.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919737) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919748) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919762) |
| base-b-test-8-npu-a3 / run (0) | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919765) |
| base-b-test-4-npu-a3 / run (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919783) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919835) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919884) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919940) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919965) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919967) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874417154/job/94987919971) |


## [Run #31874378798](https://github.com/sgl-project/sglang/actions/runs/31874378798)
- **分支**: `voidc-minor/jit-moe-topk-softmax`
- **总耗时**: 87.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31874378798

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.1min | 超时 | 性能测试用例执行超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94988610769) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94991756409) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94992160267) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94996277015) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1170秒后退出码1，未通过，导致整个作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94988610769

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败机制跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94991756409

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94992160267

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94996277015

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987842824) |
| base-b-test-2-npu-a3 / run (0) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987842828) |
| base-b-test-4-npu-a3 / run (0) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987842856) |
| base-b-test-16-npu-a3 / run (0) | 61.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987842861) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987842864) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987842887) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987842948) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987843051) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987843060) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987843065) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31874378798/job/94987843117) |


## [Run #31873761013](https://github.com/sgl-project/sglang/actions/runs/31873761013)
- **分支**: `codex/k3-mm-cache-k3`
- **总耗时**: 100.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31873761013

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293459) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未达预期，测试用例执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986767222) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94990175463) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94991294585) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 作业因其他根因作业失败而被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94994976186) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。作业最终上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293459

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1127秒，未通过性能基准，可能因模型推理速度或吞吐量未达标导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986767222

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业触发了fast-fail，导致本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94990175463

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，本作业被Fast-fail机制跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94991294585

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3，本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94994976186

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293475) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293506) |
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293554) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293573) |
| base-b-test-16-npu-a3 / run (0) | 37.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293595) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293598) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293610) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293643) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293700) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 37.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293714) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873761013/job/94986293805) |


## [Run #31873565285](https://github.com/sgl-project/sglang/actions/runs/31873565285)
- **分支**: `pllimax/output-log-dir-structure`
- **总耗时**: 28.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31873565285

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 1.4min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764737) |
| base-a-test-1-npu-a2 / run (0) | 4.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764739) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764781) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现多个根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764843) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.3min | 环境问题 | 测试脚本启动后立即失败，退出码1，无具体测试日志 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764945) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | 环境问题 | 测试启动后立即失败，无具体测试日志，疑似环境或基础设施问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764976) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.2min | 环境问题 | 测试脚本启动后立即失败，退出码1，无具体测试日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764996) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因同批次其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985765074) |

- **base-b-test-4-npu-a3 / run (1)**: 日志显示本作业在健康检查阶段因检测到base-c-test-acc-2-npu-a3和base-c-test-acc-8-npu-a3两个根因作业失败而被快速失败（fast-fail），并非自身测试失败，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764737

- **base-a-test-1-npu-a2 / run (0)**: 本作业在健康检查阶段检测到base-c-test-acc-2/8/16-npu-a3等根因作业已失败，按策略快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764739

- **base-b-test-4-npu-a3 / run (0)**: 健康检查发现根因作业base-c-test-acc-8-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764781

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了级联失败后，根因失败作业为base-c-test-acc系列（2/8/16-npu-a3），本作业因快速失败被取消，并非自身代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764843

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在运行测试命令时立即以退出码1终止，日志中未显示任何测试输出或错误详情，可能是NPU环境初始化失败、容器配置问题或依赖安装不完整导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764945

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在运行测试脚本时立即以退出码1终止，错误信息仅显示“command terminated with non-zero exit code”，未提供具体测试输出，可能因NPU环境配置、资源分配或容器启动问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764976

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在运行测试命令时立即报错退出（exit code 1），日志中未显示任何测试用例执行信息，可能因环境初始化失败、依赖缺失或容器问题导致，需查看完整日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764996

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业base-c-test-acc-8-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于CI流程的级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985765074

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764753) |
| base-b-test-1-npu-a3 / run (0) | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764792) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873565285/job/94985764835) |


## [Run #31873437889](https://github.com/sgl-project/sglang/actions/runs/31873437889)
- **分支**: `mmangkad/fix-flashinfer-allreduce-workspace-warning`
- **总耗时**: 300.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31873437889

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.7min | 性能回归 | NPU性能测试未通过，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985886684) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因同PR中另一作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94989818302) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94993613694) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1137秒后失败，0/1测试通过，属于性能测试未达标或执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985886684

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示health-check检测到同PR的base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖导致的跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94989818302

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业，导致本作业被快速失败跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94993613694

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985442890) |
| base-b-test-16-npu-a3 / run (0) | 53.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985442904) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985442907) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985442951) |
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985442957) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985442964) |
| base-a-test-1-npu-a2 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985443008) |
| base-b-test-4-npu-a3 / run (0) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985443048) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985443057) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 79.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985443059) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985443085) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94985443174) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 265.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31873437889/job/94988009012) |


## [Run #31872408774](https://github.com/sgl-project/sglang/actions/runs/31872408774)
- **分支**: `main`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31872408774

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31872408774/job/94983034602) |

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31872408774/job/94983034602


## [Run #31871829383](https://github.com/sgl-project/sglang/actions/runs/31871829383)
- **分支**: `wca-rebased`
- **总耗时**: 115.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31871829383

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.5min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94982946692) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现同PR中其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94984864195) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94986773275) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业（8-NPU）已失败，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94993291572) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1081秒后失败，该测试为性能测试，可能因推理性能未达到预期阈值（如50ms）而判定失败，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94982946692

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到同PR的base-c-test-perf-8-npu-a3作业失败，根据快速失败策略，本作业被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94984864195

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在启动阶段即被健康检查机制终止，原因是同一次运行中base-c-test-perf-8-npu-a3作业失败，触发了fast-fail逻辑，本作业未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94986773275

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94993291572

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 46.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694274) |
| multimodal-gen-test-1-npu-a3 | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694296) |
| base-b-test-4-npu-a3 / run (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694317) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694323) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694335) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694365) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694368) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694434) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694453) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694454) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694523) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871829383/job/94981694576) |


## [Run #31871531794](https://github.com/sgl-project/sglang/actions/runs/31871531794)
- **分支**: `main_8.5`
- **总耗时**: 8.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31871531794

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.4min | 其他 | 作业日志被截断，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865433) |
| base-b-test-8-npu-a3 / run (0) | 4.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865488) |
| base-b-test-16-npu-a3 / run (0) | 6.2min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865501) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | 代码错误 | NPU注意力测试test_npu_ascend_dsv4_backend.py失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865530) |
| base-b-test-1-npu-a3 / run (0) | 6.4min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865531) |
| base-b-test-2-npu-a3 / run (0) | 6.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865535) |
| base-b-test-4-npu-a3 / run (0) | 4.9min | 环境问题 | 自定义容器执行失败，NPU图捕获过程中容器异常退出 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865569) |
| base-b-test-4-npu-a3 / run (1) | 0.6min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865647) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 环境问题 | 自定义容器启动失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865730) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.6min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865766) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865775) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.7min | 环境问题 | 自定义容器执行失败，导致作业在启动阶段中止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865796) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示上传artifact时未找到diffusion-failures目录，说明测试可能未产生失败文件，但无法确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865433

- **base-b-test-8-npu-a3 / run (0)**: 测试开始后约23秒，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU容器启动或初始化问题，导致测试无法正常运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865488

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载到50%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865501

- **base-a-test-1-npu-a2 / run (0)**: 测试test_npu_ascend_dsv4_backend.py执行失败（exit code 1），而test_npu_ascend_backend.py通过。可能是DSV4后端相关代码存在bug或兼容性问题，需检查该测试文件及对应实现。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865530

- **base-b-test-1-npu-a3 / run (0)**: 测试运行到第2个文件时，自定义容器实现执行失败，runner提示联系管理员，可能是容器环境或资源问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865531

- **base-b-test-2-npu-a3 / run (0)**: 测试运行到expert distribution记录后，自定义容器实现执行失败，提示联系runner管理员，属于NPU自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865535

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在CUDA图捕获进行到14%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU环境不稳定或容器资源限制导致，属于基础设施问题而非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865569

- **base-b-test-4-npu-a3 / run (1)**: 日志显示runner在运行k8s/index.js时出现错误，提示联系自托管runner管理员，属于基础设施或容器环境配置问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865647

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动自定义容器时失败，错误信息为'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于runner或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865730

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示执行自定义容器实现时出错，提示联系自托管runner管理员。可能是镜像拉取失败、容器配置错误或runner环境异常，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865766

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865775

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在安装依赖后执行自定义容器时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871531794/job/94980865796


## [Run #31871506669](https://github.com/sgl-project/sglang/actions/runs/31871506669)
- **分支**: `main`
- **总耗时**: 20.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31871506669

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 19.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844396) |
| base-b-test-4-npu-a3 / run (0) | 8.5min | 超时 | HiCache MLA测试超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844411) |
| base-b-test-4-npu-a3 / run (1) | 18.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844417) |
| base-b-test-1-npu-a3 / run (0) | 19.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844425) |
| multimodal-gen-test-1-npu-a3 | 19.0min | 环境问题 | 作业因缺少diffusion-failures目录而失败，但未显示具体测试错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844430) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844437) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 18.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844487) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 18.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844535) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 18.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844565) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94981271075) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844396

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py运行301秒后超时退出（预计400秒），返回码1，导致整个作业失败。测试未通过，可能因NPU性能或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844411

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型权重加载到75%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境基础设施问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844417

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中容器突然报错'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于NPU CI环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844425

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures工件时未找到文件，说明测试未生成失败样本。作业可能因测试未运行或提前退出而失败，但日志未提供具体错误信息，需进一步查看测试步骤日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844430

- **base-b-test-2-npu-a3 / run (0)**: 作业在DeepseekV2模型权重加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844437

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844487

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在执行过程中自定义容器实现失败，提示联系runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844535

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行中突然出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844565

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到根因作业 base-b-test-4-npu-a3 / run (0) 失败，因此本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94981271075

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844409) |
| base-b-test-8-npu-a3 / run (0) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844433) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871506669/job/94980844548) |


## [Run #31871467376](https://github.com/sgl-project/sglang/actions/runs/31871467376)
- **分支**: `case-tpot-fix`
- **总耗时**: 208.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31871467376

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024800) |
| base-b-test-4-npu-a3 / run (1) | 13.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024813) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024826) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024853) |
| base-b-test-16-npu-a3 / run (0) | 49.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024861) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024876) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024879) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024905) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024917) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024957) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 128.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983024960) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94983430824) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94985691067) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94986830117) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 75.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871467376/job/94996007295) |


## [Run #31871115743](https://github.com/sgl-project/sglang/actions/runs/31871115743)
- **分支**: `cheng/gc-sr-review`
- **总耗时**: 20.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31871115743

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449423) |
| base-b-test-1-npu-a3 / run (0) | 18.9min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449465) |
| base-b-test-4-npu-a3 / run (0) | 17.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449490) |
| base-b-test-2-npu-a3 / run (0) | 17.4min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449496) |
| base-b-test-16-npu-a3 / run (0) | 17.8min | 超时 | TokenizerManager watchdog超时导致容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449536) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449627) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449633) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 16.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449668) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 4.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94982110150) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449423

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在加载模型权重时（Multi-thread loading shards 0%）容器实现执行失败，错误为自定义容器实现问题，属于NPU CI环境或容器配置异常，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449465

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449490

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重时（约50%进度）自定义容器实现执行失败，提示联系自托管runner管理员。日志显示存在vllm/mindspore等模块缺失及mm_utils导入错误，但核心原因是容器运行环境不稳定。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449496

- **base-b-test-16-npu-a3 / run (0)**: 日志显示TokenizerManager watchdog超时（300秒），服务启动过程中卡死，最终导致自定义容器执行失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449536

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行约17分钟后，在Decode阶段正常输出时突然报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449627

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但中途出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449633

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在执行过程中出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449668

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系runner管理员，属于NPU CI环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94982110150

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449473) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449555) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449606) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31871115743/job/94981449667) |


## [Run #31871114383](https://github.com/sgl-project/sglang/actions/runs/31871114383)
- **分支**: `cheng/gc-sr-5-debt-ratchet`
- **总耗时**: 5.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31871114383

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| check-changes | 5.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31871114383/job/94979763315) |

- **check-changes**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31871114383/job/94979763315


---
*Auto-generated by npu_pr_monitor.py*