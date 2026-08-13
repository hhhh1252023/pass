# NPU CI 执行监控
**生成时间**: 2026-08-13 12:22 UTC
**分析 Run 数**: 39

---

## 📊 本次执行总结

- **成功 Job 数**: 49
- **失败 Run 数**: 34
- **成功 Job 平均耗时**: 28.4min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 108.2min | #31675717950 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809779) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.7min | #31675757749 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922731) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 78.8min | #31675717950 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94418797948) |
| base-b-test-16-npu-a3 / run (0) | 65.6min | #31675717950 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809540) |
| base-b-test-16-npu-a3 / run (0) | 50.9min | #31675757749 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922463) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 49.2min | #31675717950 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809836) |
| multimodal-gen-test-1-npu-a3 | 42.9min | #31676807013 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676807013/job/94373129159) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.0min | #31675757749 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922751) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.3min | #31675717950 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809851) |
| multimodal-gen-test-1-npu-a3 | 37.0min | #31675717950 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809490) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 结论 | 链接 |
|--------|------|------|-----------|------|------|
| #31675717950<br>[#33165 [AMD] DeepSeek-V4 MI355X: eliminate bpreshuffle fp8-scale relayout copy in dense w8a8 linear](https://github.com/sgl-project/sglang/pull/33165) | `pr/dsv4-bpreshuffle-scale-nocopy-dense` | 292.0min | 1 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31675717950) |
| #31675757749<br>[#32340 Amd/dsv4 shared experts fusion top6](https://github.com/sgl-project/sglang/pull/32340) | `amd/dsv4-shared-experts-fusion-top6` | 229.7min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31675757749) |
| #31676677882<br>[#34542 [MiniMax-M3] Overlap shared and routed experts](https://github.com/sgl-project/sglang/pull/34542) | `minimax-m3-moe-dual-stream` | 222.3min | 11 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31676677882) |
| #31676290697<br>[#34492 XPU: SGLANG_USE_SGL_XPU default to true](https://github.com/sgl-project/sglang/pull/34492) | `SGLANG_USE_SGL_XPU` | 209.1min | 11 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31676290697) |
| #31687601256<br>[#34616 [Diffusion][FLUX.2] Fuse eager AdaLN and packed SwiGLU](https://github.com/sgl-project/sglang/pull/34616) | `bbuf/b300-flux2-eager-fusions` | 136.6min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31687601256) |
| #31687604786<br>[#34617 [Diffusion][HunyuanVideo] Fuse eager QKV packing and high-quality QKNorm](https://github.com/sgl-project/sglang/pull/34617) | `bbuf/b300-hunyuan-eager-fusions` | 134.7min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31687604786) |
| #31681202432<br>[#28932 [AMD] Add dense-FP8 for MXFP4 checkpoints with fused silu, mul, activation quant](https://github.com/sgl-project/sglang/pull/28932) | `marv/fuse_down_proj_act_quant_silu_mul` | 124.0min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31681202432) |
| #31685479559<br>[#34565 [Unified Tree] Support Branching-Point Caching for the SWA Component](https://github.com/sgl-project/sglang/pull/34565) | `cjc/unified-swa-branching` | 98.2min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31685479559) |
| #31679729610<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 76.7min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31679729610) |
| #31679684931<br>[#34597 [AMD] Run V4 MTP target-verify through the decode kernel](https://github.com/sgl-project/sglang/pull/34597) | `v4-mtp-verify-kernel` | 67.4min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31679684931) |
| #31680815323<br>[#34565 [Unified Tree] Support Branching-Point Caching for the SWA Component](https://github.com/sgl-project/sglang/pull/34565) | `cjc/unified-swa-branching` | 62.5min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31680815323) |
| #31678307473 | `rainj-me/rust-server-pd-lb` | 56.5min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31678307473) |
| #31686621503<br>[#34584 [diffusion] Wan2.2-TI2V: fuse per-token adaLN table add into contiguous slices + hoist rope cache (denoise -13.1% H100 / -12.6% H200, bit-exact; eager beats compile)](https://github.com/sgl-project/sglang/pull/34584) | `main` | 54.3min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31686621503) |
| #31686024994<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 48.7min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31686024994) |
| #31682372905 | `rainj-me/rust-server-pd-lb` | 43.6min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31682372905) |
| #31676991923<br>[#32327 [DeepSeek-V4] Add Q8KV8 sparse MLA prefill runtime backend](https://github.com/sgl-project/sglang/pull/32327) | `dsv4/mhc-q8kv8-prefill` | 40.1min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31676991923) |
| #31676413870<br>[#33030 [NPU] add Ascend 950 (Atlas A5) backend paths for DeepSeek-V4](https://github.com/sgl-project/sglang/pull/33030) | `dsv4_a5_pr` | 36.5min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31676413870) |
| #31694723229<br>[#32637 Optimize delayed sample and mrope position computation](https://github.com/sgl-project/sglang/pull/32637) | `main` | 33.0min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31694723229) |
| #31692891052<br>[#34713 [diffusion] Decouple encoder parallelism from the DiT parallel layout](https://github.com/sgl-project/sglang/pull/34713) | `encoder-dp-under-tp` | 31.1min | 1 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31692891052) |
| #31688870566<br>[#34713 [diffusion] Decouple encoder parallelism from the DiT parallel layout](https://github.com/sgl-project/sglang/pull/34713) | `encoder-dp-under-tp` | 31.1min | 1 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31688870566) |
| #31680489948<br>[#33907 [Perf] Free out-of-window SWA pages without a device sync](https://github.com/sgl-project/sglang/pull/33907) | `lsyin/swa-sync-free` | 27.6min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31680489948) |
| #31679490404<br>[#29593 [CPU][QUANT] add amx cpu support for auto-round](https://github.com/sgl-project/sglang/pull/29593) | `main` | 24.8min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31679490404) |
| #31677237743<br>[#34421 [AMD][Perf] Fuse GatedDeltaNet QKVZBA split/reshape/cat into a single Triton kernel for Qwen3.5-architecture MoE on HIP](https://github.com/sgl-project/sglang/pull/34421) | `main` | 21.9min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31677237743) |
| #31678143534 | `feature/load-reporter` | 18.1min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31678143534) |
| #31695659121 | `fix/qwen35-hicache-mtp-draft-depth` | 15.8min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31695659121) |
| #31681269015<br>[#32755  [Perf] Occupancy tuning for DSA indexer fp8-quant Q kernel](https://github.com/sgl-project/sglang/pull/32755) | `main` | 15.2min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31681269015) |
| #31691150705<br>[#34713 [diffusion] Decouple encoder parallelism from the DiT parallel layout](https://github.com/sgl-project/sglang/pull/34713) | `encoder-dp-under-tp` | 15.1min | 1 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31691150705) |
| #31687594851<br>[#34615 [Diffusion] Make auto residency decisions component-scoped](https://github.com/sgl-project/sglang/pull/34615) | `bbuf/b300-flux2-residency` | 14.0min | 1 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31687594851) |
| #31684646681<br>[#34597 [AMD] Run V4 MTP target-verify through the decode kernel](https://github.com/sgl-project/sglang/pull/34597) | `main` | 13.2min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31684646681) |
| #31678717575<br>[#30762 fix(hicache/umbp): support DeepSeek-V4 hybrid HostPoolGroup (multi-po…](https://github.com/sgl-project/sglang/pull/30762) | `main` | 10.5min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31678717575) |
| #31692255269<br>[#34713 [diffusion] Decouple encoder parallelism from the DiT parallel layout](https://github.com/sgl-project/sglang/pull/34713) | `encoder-dp-under-tp` | 9.0min | 1 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31692255269) |
| #31676270238<br>[#31956 Optimize MiniMax-M2.7 on CPU](https://github.com/sgl-project/sglang/pull/31956) | `main` | 8.6min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31676270238) |
| #31685419623<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 8.4min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31685419623) |
| #31675899335<br>[#32991 feat(attention): add architecture-owned SM12x FA4 kernels](https://github.com/sgl-project/sglang/pull/32991) | `main` | 5.9min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31675899335) |

---


## [Run #31695659121](https://github.com/sgl-project/sglang/actions/runs/31695659121)
- **分支**: `fix/qwen35-hicache-mtp-draft-depth`
- **总耗时**: 15.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31695659121

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.4min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832006) |
| base-b-test-1-npu-a3 / run (0) | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832177) |
| base-b-test-16-npu-a3 / run (0) | 14.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832212) |
| base-a-test-1-npu-a2 / run (0) | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832235) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832255) |
| base-b-test-2-npu-a3 / run (0) | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832282) |
| base-b-test-4-npu-a3 / run (0) | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832327) |
| base-b-test-8-npu-a3 / run (0) | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832335) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832532) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832598) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832685) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832755) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅显示Node 20弃用警告和上传diffusion-failures目录时未找到文件。可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832006

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832177

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832212

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源是否存在或更新下载链接。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832235

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件已被删除或路径错误，可能是资源清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832255

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832282

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832327

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832335

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832532

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832598

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832685

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31695659121/job/94432832755


## [Run #31694723229](https://github.com/sgl-project/sglang/actions/runs/31694723229)
- **分支**: `main`
- **总耗时**: 33.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31694723229

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.4min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847718) |
| base-b-test-1-npu-a3 / run (0) | 32.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847906) |
| base-a-test-1-npu-a2 / run (0) | 32.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847935) |
| base-b-test-4-npu-a3 / run (0) | 32.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847936) |
| base-b-test-2-npu-a3 / run (0) | 32.3min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847940) |
| base-b-test-16-npu-a3 / run (0) | 32.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847956) |
| base-b-test-4-npu-a3 / run (1) | 32.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848000) |
| base-b-test-8-npu-a3 / run (0) | 32.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848066) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 32.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848322) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 32.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848346) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 32.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848363) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 32.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848475) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847718

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847906

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847935

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847936

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847940

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖缺失或配置错误，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429847956

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848000

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848066

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848322

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848346

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848363

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31694723229/job/94429848475


## [Run #31692891052](https://github.com/sgl-project/sglang/actions/runs/31692891052)
- **分支**: `encoder-dp-under-tp`
- **总耗时**: 31.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31692891052

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 29.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31692891052/job/94424176320) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未包含多模态生成测试的具体执行步骤或错误输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31692891052/job/94424176320


## [Run #31692255269](https://github.com/sgl-project/sglang/actions/runs/31692255269)
- **分支**: `encoder-dp-under-tp`
- **总耗时**: 9.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31692255269

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.7min | 环境问题 | 作业因缺少diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31692255269/job/94422130402) |

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures工件时未找到文件，说明测试未产生失败样本，可能因环境配置或前置步骤异常导致测试未运行，而非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31692255269/job/94422130402


## [Run #31691150705](https://github.com/sgl-project/sglang/actions/runs/31691150705)
- **分支**: `encoder-dp-under-tp`
- **总耗时**: 15.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31691150705

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31691150705/job/94418659555) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31691150705/job/94418659555


## [Run #31688870566](https://github.com/sgl-project/sglang/actions/runs/31688870566)
- **分支**: `encoder-dp-under-tp`
- **总耗时**: 31.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31688870566

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31688870566/job/94411375730) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未展示multimodal-gen测试的具体执行过程或错误输出，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31688870566/job/94411375730


## [Run #31688628012](https://github.com/sgl-project/sglang/actions/runs/31688628012)
- **分支**: `bbuf/b300-flux2-residency`
- **总耗时**: 46.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31688628012

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31688628012/job/94410643012) |


## [Run #31687604786](https://github.com/sgl-project/sglang/actions/runs/31687604786)
- **分支**: `bbuf/b300-hunyuan-eager-fusions`
- **总耗时**: 134.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31687604786

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 134.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318900) |
| base-b-test-8-npu-a3 / run (0) | 134.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318915) |
| base-b-test-4-npu-a3 / run (1) | 134.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318933) |
| base-b-test-4-npu-a3 / run (0) | 134.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318962) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | 环境问题 | rustup 下载中断导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318995) |
| base-b-test-1-npu-a3 / run (0) | 134.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319093) |
| base-b-test-2-npu-a3 / run (0) | 134.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319192) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 134.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319625) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 134.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319654) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 134.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319771) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 134.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407320067) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318900

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，可能是资源清理或配置问题，需检查相关 blob 路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318915

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318933

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318962

- **base-a-test-1-npu-a2 / run (0)**: 安装 Rust 时从缓存服务器下载 rustup-init 中途连接断开（curl 错误 18），剩余约 16MB 未下载完成，导致脚本退出码 18，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318995

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319093

- **base-b-test-2-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，表明CI流程尝试访问的Azure Blob存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319192

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319625

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径配置错误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319654

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407319771

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407320067

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31687604786/job/94407318555) |


## [Run #31687601256](https://github.com/sgl-project/sglang/actions/runs/31687601256)
- **分支**: `bbuf/b300-flux2-eager-fusions`
- **总耗时**: 136.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31687601256

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316816) |
| base-b-test-16-npu-a3 / run (0) | 136.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316849) |
| base-b-test-4-npu-a3 / run (1) | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316883) |
| base-b-test-1-npu-a3 / run (0) | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316891) |
| base-b-test-8-npu-a3 / run (0) | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316989) |
| base-b-test-4-npu-a3 / run (0) | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317126) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317265) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317319) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317428) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 136.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317431) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316816

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316849

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源（如模型权重、数据集或缓存）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316883

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316891

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316989

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317126

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317265

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在 Azure Blob 存储中找到，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317319

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317428

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407317431

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316606) |
| base-a-test-1-npu-a2 / run (0) | 12.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31687601256/job/94407316805) |


## [Run #31687594851](https://github.com/sgl-project/sglang/actions/runs/31687594851)
- **分支**: `bbuf/b300-flux2-residency`
- **总耗时**: 14.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31687594851

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31687594851/job/94407320880) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31687594851/job/94407320880


## [Run #31686621503](https://github.com/sgl-project/sglang/actions/runs/31686621503)
- **分支**: `main`
- **总耗时**: 54.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31686621503

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | 环境问题 | 自托管runner在安装rustup时下载超时，导致容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178888) |
| base-b-test-8-npu-a3 / run (0) | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178906) |
| base-b-test-16-npu-a3 / run (0) | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178947) |
| base-b-test-1-npu-a3 / run (0) | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178960) |
| base-b-test-4-npu-a3 / run (1) | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179025) |
| base-b-test-2-npu-a3 / run (0) | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179067) |
| base-b-test-4-npu-a3 / run (0) | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179078) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179286) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179327) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179414) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 53.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179435) |

- **base-a-test-1-npu-a2 / run (0)**: 日志显示runner尝试从内部缓存服务下载rustup-init，但长时间无响应（约4分钟），最终报错“Executing the custom container implementation failed”，属于网络或缓存服务故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178888

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178906

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178947

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178960

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179025

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是上游构建未成功上传或存储配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179067

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179078

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179286

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179327

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179414

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404179435

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31686621503/job/94404178844) |


## [Run #31686024994](https://github.com/sgl-project/sglang/actions/runs/31686024994)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 48.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31686024994

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 47.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327051) |
| base-a-test-1-npu-a2 / run (0) | 47.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327053) |
| base-b-test-16-npu-a3 / run (0) | 47.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327134) |
| base-b-test-2-npu-a3 / run (0) | 47.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327150) |
| base-b-test-4-npu-a3 / run (1) | 47.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327187) |
| base-b-test-4-npu-a3 / run (0) | 47.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327196) |
| base-b-test-8-npu-a3 / run (0) | 47.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327308) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 47.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327530) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 47.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327576) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 47.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327612) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 47.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327716) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327051

- **base-a-test-1-npu-a2 / run (0)**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是CI配置中引用的文件被删除、路径错误或存储账户问题，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327053

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/日志文件在存储中缺失，可能是由于上游作业未成功上传或存储被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327134

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327150

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327187

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327196

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327308

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327530

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327576

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327612

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402327716

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31686024994/job/94402326797) |


## [Run #31685479559](https://github.com/sgl-project/sglang/actions/runs/31685479559)
- **分支**: `cjc/unified-swa-branching`
- **总耗时**: 98.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31685479559

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 97.5min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400537894) |
| base-b-test-4-npu-a3 / run (0) | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538013) |
| base-b-test-4-npu-a3 / run (1) | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538047) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | 环境问题 | rustup 下载中断导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538065) |
| base-b-test-2-npu-a3 / run (0) | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538066) |
| base-b-test-8-npu-a3 / run (0) | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538100) |
| base-b-test-1-npu-a3 / run (0) | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538251) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538413) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538445) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538479) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 97.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538547) |

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载的blob（可能为测试数据或构建产物）已被删除或路径错误，属于基础设施配置或资源缺失问题，需检查相关存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400537894

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538013

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538047

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 时，从内部缓存服务器下载 rustup-init 过程中连接中断（curl 错误 18），剩余约 16MB 未下载完成，导致脚本退出码 18，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538065

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538066

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538100

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538251

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538413

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538445

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538479

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或被误删，需检查存储配置及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400538547

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31685479559/job/94400537836) |


## [Run #31685419623](https://github.com/sgl-project/sglang/actions/runs/31685419623)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 8.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31685419623

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.9min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332207) |
| base-b-test-2-npu-a3 / run (0) | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332262) |
| base-b-test-16-npu-a3 / run (0) | 7.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332266) |
| base-b-test-1-npu-a3 / run (0) | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332273) |
| base-b-test-4-npu-a3 / run (0) | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332287) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332311) |
| base-b-test-4-npu-a3 / run (1) | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332389) |
| base-a-test-1-npu-a2 / run (0) | 7.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332396) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332702) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332704) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332774) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332901) |

- **multimodal-gen-test-1-npu-a3**: 日志截断，缺少测试执行阶段输出，无法判断失败原因。仅见Node 20弃用警告及上传失败产物（无文件），需查看完整日志定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332207

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332262

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是因为日志文件未生成、被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332266

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332273

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332287

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/缓存文件在存储中缺失，可能是由于文件被清理、路径错误或上传失败，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332311

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332389

- **base-a-test-1-npu-a2 / run (0)**: 日志返回 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332396

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332702

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332704

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332774

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31685419623/job/94400332901


## [Run #31684646681](https://github.com/sgl-project/sglang/actions/runs/31684646681)
- **分支**: `main`
- **总耗时**: 13.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31684646681

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832493) |
| base-a-test-1-npu-a2 / run (0) | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832558) |
| base-b-test-4-npu-a3 / run (1) | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832622) |
| base-b-test-4-npu-a3 / run (0) | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832693) |
| base-b-test-1-npu-a3 / run (0) | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832723) |
| base-b-test-16-npu-a3 / run (0) | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832876) |
| base-b-test-8-npu-a3 / run (0) | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832891) |
| base-b-test-2-npu-a3 / run (0) | 12.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833003) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833058) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833101) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833168) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833234) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能因资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832493

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，可能是由于过期清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832558

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832622

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832693

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存、依赖或上传文件未正确生成，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832723

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832876

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397832891

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833003

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833058

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833101

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833168

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31684646681/job/94397833234


## [Run #31683443330](https://github.com/sgl-project/sglang/actions/runs/31683443330)
- **分支**: `cosmos3_guardrails_npu`
- **总耗时**: 46.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31683443330

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31683443330/job/94409936601) |


## [Run #31682372905](https://github.com/sgl-project/sglang/actions/runs/31682372905)
- **分支**: `rainj-me/rust-server-pd-lb`
- **总耗时**: 43.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31682372905

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691582) |
| base-b-test-8-npu-a3 / run (0) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691721) |
| base-a-test-1-npu-a2 / run (0) | 1.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691737) |
| base-b-test-1-npu-a3 / run (0) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691806) |
| base-b-test-4-npu-a3 / run (1) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691830) |
| base-b-test-4-npu-a3 / run (0) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691861) |
| base-b-test-2-npu-a3 / run (0) | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691899) |
| base-b-test-16-npu-a3 / run (0) | 42.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691907) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390692317) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390692374) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390692469) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 42.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390692498) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691582

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691721

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691737

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691806

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691830

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691861

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691899

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390691907

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390692317

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390692374

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390692469

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31682372905/job/94390692498


## [Run #31681269015](https://github.com/sgl-project/sglang/actions/runs/31681269015)
- **分支**: `main`
- **总耗时**: 15.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31681269015

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141009) |
| base-b-test-16-npu-a3 / run (0) | 14.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141026) |
| base-b-test-8-npu-a3 / run (0) | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141046) |
| base-b-test-4-npu-a3 / run (0) | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141096) |
| base-b-test-2-npu-a3 / run (0) | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141149) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141164) |
| base-a-test-1-npu-a2 / run (0) | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141175) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141566) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141570) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141599) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141908) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141009

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载日志文件，但返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施/存储配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141026

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141046

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141096

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141149

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141164

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141175

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141566

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141570

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141599

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681269015/job/94387141908


## [Run #31681202432](https://github.com/sgl-project/sglang/actions/runs/31681202432)
- **分支**: `marv/fuse_down_proj_act_quant_silu_mul`
- **总耗时**: 124.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31681202432

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 3.8min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920046) |
| base-b-test-8-npu-a3 / run (0) | 123.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920114) |
| base-b-test-1-npu-a3 / run (0) | 123.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920143) |
| base-b-test-2-npu-a3 / run (0) | 123.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920160) |
| base-b-test-4-npu-a3 / run (0) | 123.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920198) |
| base-b-test-4-npu-a3 / run (1) | 123.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920235) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 123.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920471) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 123.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920611) |
| base-b-test-16-npu-a3 / run (0) | 123.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920905) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 123.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386921450) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 123.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386921462) |

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920046

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920114

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境或资源缺失问题，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920143

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的远程资源（如模型权重或缓存文件）已被删除或路径错误，属于环境配置或资源缺失问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920160

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查上传步骤或存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920198

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920235

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920471

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920611

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386920905

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386921450

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386921462

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31681202432/job/94386919847) |


## [Run #31680815323](https://github.com/sgl-project/sglang/actions/runs/31680815323)
- **分支**: `cjc/unified-swa-branching`
- **总耗时**: 62.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31680815323

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 61.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735315) |
| base-b-test-8-npu-a3 / run (0) | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735380) |
| base-b-test-4-npu-a3 / run (1) | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735399) |
| base-b-test-1-npu-a3 / run (0) | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735476) |
| base-b-test-2-npu-a3 / run (0) | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735479) |
| base-b-test-4-npu-a3 / run (0) | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735593) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735826) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735845) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735873) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735958) |

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735315

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的工件/文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735380

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735399

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境或资源配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735476

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735479

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735593

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735826

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或被误删，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735845

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的预构建产物或缓存文件未上传或已被删除，需检查相关存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735873

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件，属于环境配置或资源可用性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735958

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735266) |
| base-a-test-1-npu-a2 / run (0) | 9.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31680815323/job/94385735425) |


## [Run #31680489948](https://github.com/sgl-project/sglang/actions/runs/31680489948)
- **分支**: `lsyin/swa-sync-free`
- **总耗时**: 27.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31680489948

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 26.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709553) |
| multimodal-gen-test-1-npu-a3 | 26.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709598) |
| base-b-test-16-npu-a3 / run (0) | 26.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709636) |
| base-b-test-1-npu-a3 / run (0) | 26.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709685) |
| base-b-test-8-npu-a3 / run (0) | 26.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709696) |
| base-b-test-2-npu-a3 / run (0) | 26.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709708) |
| base-b-test-4-npu-a3 / run (1) | 26.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709725) |
| base-b-test-4-npu-a3 / run (0) | 26.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709802) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384710004) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384710054) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 26.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384710119) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384710169) |

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure 存储中已被删除或路径错误，属于基础设施/配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709553

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未出现测试执行或失败的具体错误，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709598

- **base-b-test-16-npu-a3 / run (0)**: 作业失败原因是访问Azure Blob存储时返回BlobNotFound错误，即请求的资源不存在。这可能是由于资源被删除、路径错误或上传未完成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709636

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709685

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在存储中已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709696

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件缺失，可能是资源未上传、路径错误或存储被清理，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709708

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709725

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384709802

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384710004

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384710054

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384710119

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31680489948/job/94384710169


## [Run #31679729610](https://github.com/sgl-project/sglang/actions/runs/31679729610)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 76.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31679729610

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630656) |
| base-b-test-4-npu-a3 / run (1) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630669) |
| base-b-test-2-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630731) |
| base-b-test-8-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630732) |
| base-b-test-4-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630737) |
| base-b-test-16-npu-a3 / run (0) | 74.7min | 环境问题 | 日志下载失败，Blob不存在 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630836) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630895) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630936) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630982) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382631076) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630656

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630669

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630731

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存或依赖文件未正确上传，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630732

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630737

- **base-b-test-16-npu-a3 / run (0)**: GitHub Actions日志中显示Azure Blob存储返回BlobNotFound错误，说明日志文件已被删除或路径错误，无法获取实际作业输出，属于基础设施/存储问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630836

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630895

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630936

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或已被删除，可能是资源清理或路径配置错误所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630982

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382631076

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630611) |
| base-a-test-1-npu-a2 / run (0) | 10.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31679729610/job/94382630765) |


## [Run #31679684931](https://github.com/sgl-project/sglang/actions/runs/31679684931)
- **分支**: `v4-mtp-verify-kernel`
- **总耗时**: 67.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31679684931

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 66.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163023) |
| base-b-test-4-npu-a3 / run (1) | 66.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163038) |
| base-b-test-4-npu-a3 / run (0) | 66.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163039) |
| base-b-test-16-npu-a3 / run (0) | 66.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163043) |
| base-b-test-2-npu-a3 / run (0) | 66.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163046) |
| base-b-test-8-npu-a3 / run (0) | 66.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163105) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 66.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163363) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 66.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163431) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 66.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163439) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 66.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163447) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163023

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163038

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163039

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163043

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163046

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163105

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163363

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163431

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163439

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163447

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382162916) |
| base-a-test-1-npu-a2 / run (0) | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31679684931/job/94382163056) |


## [Run #31679552463](https://github.com/sgl-project/sglang/actions/runs/31679552463)
- **分支**: `adaln-online`
- **总耗时**: 51.1min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31679552463

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31679552463/job/94386325679) |


## [Run #31679490404](https://github.com/sgl-project/sglang/actions/runs/31679490404)
- **分支**: `main`
- **总耗时**: 24.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31679490404

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 19.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543325) |
| base-b-test-16-npu-a3 / run (0) | 24.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543360) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543393) |
| base-b-test-2-npu-a3 / run (0) | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543397) |
| base-a-test-1-npu-a2 / run (0) | 24.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543429) |
| base-b-test-4-npu-a3 / run (1) | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543445) |
| base-b-test-4-npu-a3 / run (0) | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543478) |
| base-b-test-8-npu-a3 / run (0) | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543492) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543712) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543752) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543761) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 24.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543781) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行或失败断言，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543325

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，可能是构建产物或依赖缓存缺失，需检查存储配置或重新上传对应文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543360

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543393

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543397

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543429

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源已被删除或路径错误，属于外部存储依赖问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543445

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543478

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543492

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543712

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543752

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储账户中找到，可能是文件被清理、路径错误或上传失败，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543761

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31679490404/job/94381543781


## [Run #31678717575](https://github.com/sgl-project/sglang/actions/runs/31678717575)
- **分支**: `main`
- **总耗时**: 10.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31678717575

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 9.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188693) |
| multimodal-gen-test-1-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188720) |
| base-b-test-4-npu-a3 / run (0) | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188779) |
| base-b-test-1-npu-a3 / run (0) | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188791) |
| base-b-test-8-npu-a3 / run (0) | 9.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188792) |
| base-b-test-4-npu-a3 / run (1) | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188848) |
| base-b-test-2-npu-a3 / run (0) | 9.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188905) |
| base-a-test-1-npu-a2 / run (0) | 9.2min | 环境问题 | 下载triton-ascend依赖时网络中断，导致容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188909) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188966) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379189038) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379189076) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379189115) |

- **base-b-test-16-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，表明CI系统尝试访问的存储对象缺失，可能是日志上传或下载路径配置错误，或存储内容被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188693

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188720

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188779

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188791

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，可能是构建产物或依赖缓存缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188792

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188848

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188905

- **base-a-test-1-npu-a2 / run (0)**: 在安装triton-ascend==3.2.1.dev20260530时，下载速度极慢（13.6 kB/s），连接中断后重试仍无法完成，最终容器执行失败，属于网络或镜像源不稳定导致的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188909

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379188966

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379189038

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379189076

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678717575/job/94379189115


## [Run #31678307473](https://github.com/sgl-project/sglang/actions/runs/31678307473)
- **分支**: `rainj-me/rust-server-pd-lb`
- **总耗时**: 56.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31678307473

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 54.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086690) |
| base-a-test-1-npu-a2 / run (0) | 1.4min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086735) |
| base-b-test-1-npu-a3 / run (0) | 54.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086738) |
| base-b-test-4-npu-a3 / run (1) | 54.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086750) |
| base-b-test-2-npu-a3 / run (0) | 54.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086763) |
| base-b-test-16-npu-a3 / run (0) | 54.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086787) |
| base-b-test-8-npu-a3 / run (0) | 54.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086816) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 54.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378087166) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 54.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378087195) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 54.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378087232) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 54.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378087247) |

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086690

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086735

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086738

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086750

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086763

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086787

- **base-b-test-8-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，表明CI流程尝试访问的存储对象缺失或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086816

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378087166

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378087195

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378087232

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378087247

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31678307473/job/94378086602) |


## [Run #31678143534](https://github.com/sgl-project/sglang/actions/runs/31678143534)
- **分支**: `feature/load-reporter`
- **总耗时**: 18.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31678143534

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318251) |
| base-b-test-16-npu-a3 / run (0) | 17.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318261) |
| base-b-test-4-npu-a3 / run (1) | 17.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318314) |
| base-b-test-2-npu-a3 / run (0) | 17.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318315) |
| base-b-test-8-npu-a3 / run (0) | 17.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318352) |
| base-b-test-1-npu-a3 / run (0) | 17.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318360) |
| base-b-test-4-npu-a3 / run (0) | 17.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318426) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 17.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318702) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318710) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 17.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318808) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318871) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318251

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明作业尝试访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318261

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源可用性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318314

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件缺失，可能是资源未上传、路径错误或已过期，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318315

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318352

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318360

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318426

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318702

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318710

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318808

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318871

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 9.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31678143534/job/94377318394) |


## [Run #31677237743](https://github.com/sgl-project/sglang/actions/runs/31677237743)
- **分支**: `main`
- **总耗时**: 21.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31677237743

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374515816) |
| base-b-test-4-npu-a3 / run (1) | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374515915) |
| base-b-test-2-npu-a3 / run (0) | 20.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374515930) |
| base-b-test-4-npu-a3 / run (0) | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374515973) |
| base-b-test-8-npu-a3 / run (0) | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516012) |
| base-b-test-1-npu-a3 / run (0) | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516019) |
| base-b-test-16-npu-a3 / run (0) | 20.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516024) |
| base-a-test-1-npu-a2 / run (0) | 20.4min | 环境问题 | 下载triton_ascend依赖时网络连接反复中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516115) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516322) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516328) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516347) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516348) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未展示测试执行过程或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374515816

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374515915

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或资源被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374515930

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成产物，或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374515973

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516012

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516019

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516024

- **base-a-test-1-npu-a2 / run (0)**: 在安装triton_ascend-3.2.1.dev20260530时，网络多次中断，下载速度极慢（最低6.6kB/s），多次重试后仍无法完成下载，最终导致自定义容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516115

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516322

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516328

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516347

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31677237743/job/94374516348


## [Run #31676991923](https://github.com/sgl-project/sglang/actions/runs/31676991923)
- **分支**: `dsv4/mhc-q8kv8-prefill`
- **总耗时**: 40.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31676991923

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680042) |
| base-b-test-4-npu-a3 / run (1) | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680132) |
| base-b-test-4-npu-a3 / run (0) | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680143) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680152) |
| base-b-test-1-npu-a3 / run (0) | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680186) |
| base-b-test-8-npu-a3 / run (0) | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680188) |
| base-b-test-2-npu-a3 / run (0) | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680192) |
| base-b-test-16-npu-a3 / run (0) | 39.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680197) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680409) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680412) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680431) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 39.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680434) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures工件时未找到文件，说明测试可能未产生失败记录或日志被截断，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680042

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680132

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境或资源配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680143

- **base-a-test-1-npu-a2 / run (0)**: test/registered/kernels/ops/attention/test_q8kv8_sparse_prefill_backend.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 python3 file.py -f 下静默跳过，collect_tests 抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680152

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680186

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680188

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源（如模型权重、缓存或日志）已被删除或路径错误，属于环境或资源配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680192

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680197

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680409

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680412

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680431

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的依赖或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676991923/job/94373680434


## [Run #31676886732](https://github.com/sgl-project/sglang/actions/runs/31676886732)
- **分支**: `lzl/feat/support_longcat_image`
- **总耗时**: 45.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31676886732

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 36.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31676886732/job/94373383664) |


## [Run #31676807013](https://github.com/sgl-project/sglang/actions/runs/31676807013)
- **分支**: `pr/diffusion-dit-strided-residency`
- **总耗时**: 57.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31676807013

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 42.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31676807013/job/94373129159) |


## [Run #31676677882](https://github.com/sgl-project/sglang/actions/runs/31676677882)
- **分支**: `minimax-m3-moe-dual-stream`
- **总耗时**: 222.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31676677882

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728776) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728821) |
| multimodal-gen-test-1-npu-a3 | 45.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728832) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业根因失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728849) |
| base-b-test-16-npu-a3 / run (0) | 3.5min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728852) |
| base-b-test-4-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728865) |
| base-b-test-4-npu-a3 / run (1) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728883) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.4min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372729097) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372729102) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.3min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372729144) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372729166) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728776

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段被健康检查机制拦截，检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail策略，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728821

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到具体测试命令和错误输出。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本或测试未执行到该步骤。需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728832

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728849

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728852

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728865

- **base-b-test-4-npu-a3 / run (1)**: 该作业因其他作业（multimodal-gen-test-1-npu-a3）失败而被级联取消，并非自身问题。健康检查过滤了多个级联失败后，识别出根因作业并触发fast-fail，导致本作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728883

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤掉级联失败后，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372729097

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 该作业因PR健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372729102

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示本作业在健康检查阶段因检测到根因失败作业（multimodal-gen-test-1-npu-a3）而触发快速失败，并非本作业自身问题，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372729144

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于CI依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372729166

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31676677882/job/94372728870) |


## [Run #31676413870](https://github.com/sgl-project/sglang/actions/runs/31676413870)
- **分支**: `dsv4_a5_pr`
- **总耗时**: 36.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31676413870

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 19.7min | 其他 | 作业日志被截断，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001139) |
| base-b-test-4-npu-a3 / run (0) | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001247) |
| base-b-test-1-npu-a3 / run (0) | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001305) |
| base-b-test-2-npu-a3 / run (0) | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001309) |
| base-b-test-8-npu-a3 / run (0) | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001312) |
| base-a-test-1-npu-a2 / run (0) | 2.3min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001318) |
| base-b-test-16-npu-a3 / run (0) | 35.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001357) |
| base-b-test-4-npu-a3 / run (1) | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001418) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001724) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001773) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001789) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001831) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行的具体输出。从可见内容看，作业正常启动并完成上传工件步骤（无文件上传），未发现明显错误或失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001139

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001247

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001305

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001309

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001312

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 1.92 时，从内部缓存服务下载 channel-rust-1.92.toml 超时，导致脚本退出，作业失败。属于网络或缓存服务临时故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001318

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001357

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001418

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001724

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001773

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001789

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676413870/job/94372001831


## [Run #31676290697](https://github.com/sgl-project/sglang/actions/runs/31676290697)
- **分支**: `SGLANG_USE_SGL_XPU`
- **总耗时**: 209.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31676290697

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550167) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550186) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败机制触发，跳过本作业 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550188) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现根因失败任务，导致本作业被级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550253) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 环境问题 | 健康检查发现多个NPU测试作业级联失败，根因作业为base-a-test-1-npu-a2和base-c-test-acc-16-npu-a3，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550257) |
| base-a-test-1-npu-a2 / run (0) | 32.5min | 环境问题 | pip下载triton_ascend包时网络连接中断，多次重试后仍失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550278) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550338) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.5min | 环境问题 | Kubernetes Pod 启动失败，作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550392) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550428) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550443) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550447) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到base-a-test-1-npu-a2和base-c-test-acc-16-npu-a3为根因失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550167

- **base-b-test-1-npu-a3 / run (0)**: 本作业在启动前的健康检查中检测到根因作业 base-a-test-1-npu-a2 / run (0) 已失败，因此触发 fast-fail 机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550186

- **base-b-test-4-npu-a3 / run (1)**: 日志显示本作业因其他根因作业（base-a-test-1-npu-a2、base-c-test-acc-16-npu-a3）失败而被级联跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550188

- **base-b-test-4-npu-a3 / run (0)**: PR测试健康检查识别出根因失败任务（base-a-test-1-npu-a2和base-c-test-acc-16-npu-a3），本作业作为级联失败被快速失败机制跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550253

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是base-a-test-1-npu-a2和base-c-test-acc-16-npu-a3失败，触发fast-fail机制，本作业未实际运行即被终止，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550257

- **base-a-test-1-npu-a2 / run (0)**: 从华为云镜像下载triton_ascend-3.2.1.dev20260530 whl包时，网络连接反复中断，6次尝试仅下载30.5MB/188.5MB，最终因下载不完整导致安装失败，属于网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550278

- **base-b-test-2-npu-a3 / run (0)**: 该作业在健康检查阶段因其他根因作业（base-a-test-1-npu-a2、base-c-test-acc-16-npu-a3）失败而触发快速失败，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550338

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 Pod linux-aarch64-a3-16-cn12-001-772vk-runner-gqwdt-workflow 状态为 Failed，导致作业在初始化阶段即失败，未进入测试执行。属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550392

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业 base-a-test-1-npu-a2 / run (0) 失败，触发了 fast-fail 机制，本作业未实际运行即被跳过，属于依赖的上游失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550428

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示根因失败作业为base-a-test-1-npu-a2和base-c-test-acc-16-npu-a3，本作业因级联失败被过滤跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550443

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业 base-a-test-1-npu-a2 / run (0) 失败，本作业作为级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550447

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31676290697/job/94371550189) |


## [Run #31676270238](https://github.com/sgl-project/sglang/actions/runs/31676270238)
- **分支**: `main`
- **总耗时**: 8.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31676270238

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564400) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564537) |
| base-b-test-16-npu-a3 / run (0) | 7.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564541) |
| base-a-test-1-npu-a2 / run (0) | 7.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564569) |
| base-b-test-2-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564597) |
| base-b-test-1-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564667) |
| base-b-test-4-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564681) |
| base-b-test-4-npu-a3 / run (1) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564693) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564780) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564825) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564848) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564876) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564400

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的构建产物或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564537

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564541

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在下载triton-ascend依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564569

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564597

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564667

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564681

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564693

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564780

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564825

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564848

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31676270238/job/94371564876


## [Run #31675899335](https://github.com/sgl-project/sglang/actions/runs/31675899335)
- **分支**: `main`
- **总耗时**: 5.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31675899335

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415120) |
| base-b-test-8-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415141) |
| base-b-test-4-npu-a3 / run (1) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415165) |
| base-b-test-16-npu-a3 / run (0) | 4.9min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415182) |
| base-b-test-4-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415214) |
| base-b-test-2-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415231) |
| base-b-test-1-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415233) |
| base-a-test-1-npu-a2 / run (0) | 3.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415237) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415531) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415546) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415552) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415563) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415120

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置问题，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415141

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415165

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明指定的blob已被删除或路径错误，可能是CI配置中引用了过期或错误的存储路径，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415182

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415214

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415231

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415233

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在下载triton-ascend依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415237

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415531

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在存储中缺失或路径错误，可能是资源未上传、被清理或配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415546

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415552

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675899335/job/94370415563


## [Run #31675757749](https://github.com/sgl-project/sglang/actions/runs/31675757749)
- **分支**: `amd/dsv4-shared-experts-fusion-top6`
- **总耗时**: 229.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31675757749

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94404357115) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94412691192) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94416634480) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（base-c-test-perf-8-npu-a3）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94422243035) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1127秒后退出码为1，0/1通过，属于性能指标未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94404357115

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 本作业未实际运行测试，因健康检查检测到同批次base-c-test-perf-8-npu-a3作业失败，被判定为根因作业，触发fast-fail跳过执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94412691192

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动前被PR健康检查拦截，原因是同一次运行中另一个作业（base-c-test-perf-8-npu-a3）失败，被判定为根因失败，导致本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94416634480

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的PR健康检查阶段被快速失败机制终止，原因是同批次中base-c-test-perf-8-npu-a3作业已失败，本作业被级联跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94422243035

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922375) |
| base-b-test-1-npu-a3 / run (0) | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922434) |
| multimodal-gen-test-1-npu-a3 | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922453) |
| base-b-test-16-npu-a3 / run (0) | 50.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922463) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922469) |
| base-b-test-8-npu-a3 / run (0) | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922511) |
| base-b-test-2-npu-a3 / run (0) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922530) |
| base-b-test-4-npu-a3 / run (1) | 11.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922588) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922731) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922751) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922786) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675757749/job/94369922812) |


## [Run #31675717950](https://github.com/sgl-project/sglang/actions/runs/31675717950)
- **分支**: `pr/dsv4-bpreshuffle-scale-nocopy-dense`
- **总耗时**: 292.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31675717950

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 35.8min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，未达性能预期。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94410812535) |

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例退出码1，耗时1247秒，可能因性能不达标或运行错误导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94410812535

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809490) |
| base-b-test-2-npu-a3 / run (0) | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809538) |
| base-b-test-16-npu-a3 / run (0) | 65.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809540) |
| base-b-test-1-npu-a3 / run (0) | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809551) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809573) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809595) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809617) |
| base-b-test-4-npu-a3 / run (1) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809624) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 108.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809779) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 49.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809836) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809851) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94369809889) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94397739768) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94412080284) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 78.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31675717950/job/94418797948) |


---
*Auto-generated by npu_pr_monitor.py*