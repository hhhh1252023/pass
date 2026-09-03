# NPU CI 执行监控
**生成时间**: 2026-09-03 23:11 UTC
**分析 Run 数**: 20

---

## 📊 本次执行总结

- **成功 Job 数**: 8
- **失败 Run 数**: 20
- **成功 Job 平均耗时**: 5.4min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | #33785275721 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521298) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | #33788187186 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168588) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | #33786602153 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195611) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | #33781947529 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338837) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | #33780947079 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294349) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | #33761259703 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089864) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | #33788819560 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281490) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | #33786393561 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251516) |

### ❌ 耗时最长的失败 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| multimodal-gen-test-2-npu-a3 (0) | 213.3min | #33778427912 | [job link](https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043343) |
| multimodal-gen-test-1-npu-a3 (1) | 213.3min | #33778427912 | [job link](https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043436) |
| multimodal-gen-test-2-npu-a3 (1) | 213.3min | #33778427912 | [job link](https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043442) |
| multimodal-gen-test-1-npu-a3 (0) | 213.3min | #33778427912 | [job link](https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043616) |
| multimodal-gen-test-2-npu-a3 (0) | 198.5min | #33781114104 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919315) |
| multimodal-gen-test-2-npu-a3 (1) | 198.5min | #33781114104 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919435) |
| multimodal-gen-test-1-npu-a3 (1) | 198.5min | #33781114104 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919470) |
| multimodal-gen-test-1-npu-a3 (0) | 198.5min | #33781114104 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919546) |
| multimodal-gen-test-1-npu-a3 (1) | 166.8min | #33781947529 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338548) |
| multimodal-gen-test-1-npu-a3 (0) | 166.8min | #33781947529 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338578) |

---

## 📋 各任务执行统计

*无执行失败的任务。*

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #33778427912<br>[#37809 [diffusion] Calibrate residency probes that run a single iteration and honor pipeline step minimums](https://github.com/sgl-project/sglang/pull/37809) | `fix/residency-calibration-probes` | 214.1min | 4 | multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33778427912) |
| #33781114104<br>[#36380 Cosmos3 fp8 mixed precision](https://github.com/sgl-project/sglang/pull/36380) | `cosmos3-fp8-high-precision` | 199.2min | 4 | multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33781114104) |
| #33781947529<br>[#36988 [VLM] Retire aborted disaggregated prefill results](https://github.com/sgl-project/sglang/pull/36988) | `codex/retire-aborted-disagg-prefill` | 172.6min | 10 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), multimodal-gen-test-2-npu-a3 (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33781947529) |
| #33760632805<br>[#37816 [diffusion] Compose third-party component bundles safely](https://github.com/sgl-project/sglang/pull/37816) | `codex/diffusion-day0-variants` | 157.7min | 4 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33760632805) |
| #33764826439<br>[#35335 [diffusion] Warmup-calibrated auto residency promotion in performance-mode auto](https://github.com/sgl-project/sglang/pull/35335) | `mick/diffusion-auto-residency` | 105.9min | 4 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33764826439) |
| #33765202272<br>[#37266 [diffusion] MiniMax-H3: tiered AdaLN plan cache (pinned-host tier + per-plan LRU)](https://github.com/sgl-project/sglang/pull/37266) | `main` | 104.1min | 4 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33765202272) |
| #33775827167<br>[#37816 [diffusion] Compose third-party component bundles safely](https://github.com/sgl-project/sglang/pull/37816) | `codex/diffusion-day0-variants` | 63.8min | 4 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33775827167) |
| #33758314049<br>[#37680 [Diffusion] Stream mapped weights on a shared pool: O_DIRECT reader, populator, host-memory debug aids](https://github.com/sgl-project/sglang/pull/37680) | `feat/planner-unified-memory` | 60.4min | 4 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33758314049) |
| #33761259703 | `hybrid-spec-decoding` | 41.8min | 10 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33761259703) |
| #33788187186<br>[#37381 [Unified Cache][5/N]: Integrate external linker mode end to end](https://github.com/sgl-project/sglang/pull/37381) | `main` | 40.2min | 10 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33788187186) |
| #33780947079<br>[#35751 Support GPT-OSS MXFP4 checkpoints on Intel XPU](https://github.com/sgl-project/sglang/pull/35751) | `gpt-xpu` | 36.0min | 10 | base-b-test-16-npu-a3 / run (0), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33780947079) |
| #33775089892<br>[#37809 [diffusion] Calibrate residency probes that run a single iteration and honor pipeline step minimums](https://github.com/sgl-project/sglang/pull/37809) | `fix/residency-calibration-probes` | 33.5min | 4 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33775089892) |
| #33772533252<br>[#37809 [diffusion] Calibrate residency probes that run a single iteration and honor pipeline step minimums](https://github.com/sgl-project/sglang/pull/37809) | `fix/residency-calibration-probes` | 25.7min | 4 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33772533252) |
| #33786602153<br>[#37849 Fix block-scale swizzling device placement](https://github.com/sgl-project/sglang/pull/37849) | `main` | 16.1min | 10 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33786602153) |
| #33786393561<br>[#35770 [AMD] Optimize Kimi-K3 Triton MLA prefill on gfx950](https://github.com/sgl-project/sglang/pull/35770) | `perf/kimi-k3-triton-prefill` | 13.9min | 10 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), base-b-test-8-npu-a3 / run (0), multimodal-gen-test-2-npu-a3 (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33786393561) |
| #33785275721<br>[#37844 [Cache] Forward fast prefix matching capability](https://github.com/sgl-project/sglang/pull/37844) | `main` | 11.7min | 10 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33785275721) |
| #33788819560<br>[#37567 Fix buffer-mode idle tracking and VLM memory sizing](https://github.com/sgl-project/sglang/pull/37567) | `fix/buffer-idle-vlm-memory-reserve` | 11.2min | 10 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33788819560) |
| #33764028079<br>[#37760 [CI][NPU] Fix kimi_k2_6 16p in64k perf test and dsv4-flash testcases](https://github.com/sgl-project/sglang/pull/37760) | `main` | 9.0min | 7 | base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33764028079) |
| #33775312038<br>[#37825 [Bugfix] Support K2 Horizon MoE without MoVA](https://github.com/sgl-project/sglang/pull/37825) | `main` | 8.8min | 11 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33775312038) |
| #33763934650<br>[#34722 [diffusion] [NPU] Optimize LTX-2/2.3 inference performance for NPU](https://github.com/sgl-project/sglang/pull/34722) | `ltx2` | 8.1min | 11 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33763934650) |

---


## [Run #33788819560](https://github.com/sgl-project/sglang/actions/runs/33788819560)
- **分支**: `fix/buffer-idle-vlm-memory-reserve`
- **总耗时**: 11.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33788819560

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 10.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281298) |
| multimodal-gen-test-1-npu-a3 (0) | 10.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281302) |
| multimodal-gen-test-1-npu-a3 (1) | 10.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281318) |
| multimodal-gen-test-2-npu-a3 (0) | 10.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281335) |
| base-b-test-1-npu-a3 / run (0) | 10.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281454) |
| base-b-test-4-npu-a3 / run (1) | 10.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281463) |
| base-b-test-16-npu-a3 / run (0) | 10.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281477) |
| base-b-test-8-npu-a3 / run (0) | 10.5min | 环境问题 | CI作业因Azure Blob存储中找不到指定文件而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281514) |
| base-b-test-2-npu-a3 / run (0) | 10.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281581) |
| base-b-test-4-npu-a3 / run (0) | 10.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281589) |

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281298

- **multimodal-gen-test-1-npu-a3 (0)**: 作业在下载或访问某个Azure Blob存储资源时，返回BlobNotFound错误（错误码404），说明该资源已被删除或路径错误，属于外部依赖缺失的环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281302

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281318

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob资源缺失，可能是日志或依赖文件未上传或路径错误，属于环境配置或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281335

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281454

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/存储配置问题，需检查相关 blob 路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281463

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是构建产物或依赖缓存缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281477

- **base-b-test-8-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业依赖的某个blob文件不存在或已被删除，可能是构建产物或缓存缺失，属于环境或资源问题，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281514

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281581

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施/环境配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281589

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33788819560/job/100760281490) |


## [Run #33788187186](https://github.com/sgl-project/sglang/actions/runs/33788187186)
- **分支**: `main`
- **总耗时**: 40.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33788187186

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 39.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168270) |
| multimodal-gen-test-1-npu-a3 (1) | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168301) |
| multimodal-gen-test-1-npu-a3 (0) | 39.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，属于资源缺失或路径错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168331) |
| multimodal-gen-test-2-npu-a3 (0) | 39.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168359) |
| base-b-test-4-npu-a3 / run (0) | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168502) |
| base-b-test-4-npu-a3 / run (1) | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168628) |
| base-b-test-8-npu-a3 / run (0) | 39.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168663) |
| base-b-test-1-npu-a3 / run (0) | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168734) |
| base-b-test-16-npu-a3 / run (0) | 39.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168807) |
| base-b-test-2-npu-a3 / run (0) | 39.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168883) |

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168270

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168301

- **multimodal-gen-test-1-npu-a3 (0)**: 作业失败原因是下载或访问Azure Blob存储中的文件时返回BlobNotFound错误，可能是CI配置中引用的工件或依赖文件未上传或已被删除，需检查存储路径及上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168331

- **multimodal-gen-test-2-npu-a3 (0)**: 作业在下载或访问某个Azure Blob资源时，返回BlobNotFound错误，说明该资源已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168359

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未正确上传或过期清理所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168502

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168628

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中已被删除或路径错误，属于基础设施或配置问题，需检查相关存储路径或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168663

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168734

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失或配置失效，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168807

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168883

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33788187186/job/100758168588) |


## [Run #33786602153](https://github.com/sgl-project/sglang/actions/runs/33786602153)
- **分支**: `main`
- **总耗时**: 16.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33786602153

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 14.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195324) |
| multimodal-gen-test-2-npu-a3 (0) | 14.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195383) |
| multimodal-gen-test-2-npu-a3 (1) | 14.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195391) |
| multimodal-gen-test-1-npu-a3 (0) | 14.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195409) |
| base-b-test-1-npu-a3 / run (0) | 14.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195580) |
| base-b-test-2-npu-a3 / run (0) | 14.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195606) |
| base-b-test-16-npu-a3 / run (0) | 14.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195640) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195641) |
| base-b-test-4-npu-a3 / run (0) | 14.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195751) |
| base-b-test-8-npu-a3 / run (0) | 14.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195757) |

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195324

- **multimodal-gen-test-2-npu-a3 (0)**: 作业失败原因是Azure Blob存储返回BlobNotFound错误，即作业依赖的某个文件或资源在存储中不存在，可能是资源被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195383

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195391

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195409

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的 blob 资源已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195580

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195606

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195640

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195641

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195751

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是上游构建未成功上传或存储配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195757

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33786602153/job/100753195611) |


## [Run #33786393561](https://github.com/sgl-project/sglang/actions/runs/33786393561)
- **分支**: `perf/kimi-k3-triton-prefill`
- **总耗时**: 13.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33786393561

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 13.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251337) |
| multimodal-gen-test-2-npu-a3 (1) | 13.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251380) |
| multimodal-gen-test-1-npu-a3 (0) | 13.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251440) |
| base-b-test-8-npu-a3 / run (0) | 13.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251459) |
| multimodal-gen-test-2-npu-a3 (0) | 13.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251513) |
| base-b-test-2-npu-a3 / run (0) | 13.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251521) |
| base-b-test-4-npu-a3 / run (0) | 13.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251572) |
| base-b-test-1-npu-a3 / run (0) | 13.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251607) |
| base-b-test-4-npu-a3 / run (1) | 13.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251642) |
| base-b-test-16-npu-a3 / run (0) | 13.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251658) |

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传或已被删除，属于基础设施/环境配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251337

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251380

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251440

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251459

- **multimodal-gen-test-2-npu-a3 (0)**: 作业失败原因是BlobNotFound错误，即CI流程尝试下载或访问的Azure Blob存储对象不存在，可能因资源被清理、路径错误或上传失败导致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251513

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该文件已被删除或路径错误，返回 BlobNotFound 错误。这属于 CI 基础设施或日志存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251521

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251572

- **base-b-test-1-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251607

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是资源清理、上传失败或配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251642

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传延迟、路径错误或文件被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251658

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33786393561/job/100752251516) |


## [Run #33785275721](https://github.com/sgl-project/sglang/actions/runs/33785275721)
- **分支**: `main`
- **总耗时**: 11.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33785275721

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 11.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521024) |
| multimodal-gen-test-1-npu-a3 (1) | 11.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521044) |
| multimodal-gen-test-2-npu-a3 (0) | 11.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521088) |
| multimodal-gen-test-2-npu-a3 (1) | 11.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521116) |
| base-b-test-1-npu-a3 / run (0) | 11.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521224) |
| base-b-test-2-npu-a3 / run (0) | 11.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521324) |
| base-b-test-4-npu-a3 / run (1) | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521360) |
| base-b-test-4-npu-a3 / run (0) | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521363) |
| base-b-test-16-npu-a3 / run (0) | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521385) |
| base-b-test-8-npu-a3 / run (0) | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521411) |

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob资源缺失，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521024

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521044

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，可能是上游产物未上传或存储配置变更，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521088

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的依赖文件或工件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521116

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521224

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个不存在的 Blob 资源（BlobNotFound），可能是日志上传延迟、路径错误或资源被清理，属于基础设施或配置问题，与代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521324

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521360

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521363

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521385

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521411

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33785275721/job/100748521298) |


## [Run #33781947529](https://github.com/sgl-project/sglang/actions/runs/33781947529)
- **分支**: `codex/retire-aborted-disagg-prefill`
- **总耗时**: 172.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33781947529

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 166.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338548) |
| multimodal-gen-test-1-npu-a3 (0) | 166.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338578) |
| multimodal-gen-test-2-npu-a3 (0) | 166.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338687) |
| base-b-test-8-npu-a3 / run (0) | 166.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338779) |
| base-b-test-16-npu-a3 / run (0) | 166.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338782) |
| base-b-test-1-npu-a3 / run (0) | 166.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338787) |
| base-b-test-4-npu-a3 / run (0) | 166.8min | 环境问题 | CI作业因Azure Blob存储中指定文件不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338812) |
| base-b-test-2-npu-a3 / run (0) | 166.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338846) |
| base-b-test-4-npu-a3 / run (1) | 166.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338949) |
| multimodal-gen-test-2-npu-a3 (1) | 166.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739339003) |

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338548

- **multimodal-gen-test-1-npu-a3 (0)**: 作业在下载或访问某个Azure Blob资源时，返回BlobNotFound错误（错误码404），可能是资源被删除、路径错误或未上传。属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338578

- **multimodal-gen-test-2-npu-a3 (0)**: 作业日志返回BlobNotFound错误，表明CI流程尝试访问的Azure Blob存储资源缺失或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338687

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338779

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338782

- **base-b-test-1-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338787

- **base-b-test-4-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业尝试下载的构建产物或依赖文件在存储中缺失，可能是上游任务未成功上传或文件被清理，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338812

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误导致，属于基础设施或环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338846

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338949

- **multimodal-gen-test-2-npu-a3 (1)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志文件被清理、路径错误或上传失败，属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739339003

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33781947529/job/100739338837) |


## [Run #33781114104](https://github.com/sgl-project/sglang/actions/runs/33781114104)
- **分支**: `cosmos3-fp8-high-precision`
- **总耗时**: 199.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33781114104

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (0) | 198.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919315) |
| multimodal-gen-test-2-npu-a3 (1) | 198.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919435) |
| multimodal-gen-test-1-npu-a3 (1) | 198.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919470) |
| multimodal-gen-test-1-npu-a3 (0) | 198.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919546) |

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919315

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或被误删，需检查相关存储配置和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919435

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919470

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失或配置错误，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33781114104/job/100734919546


## [Run #33780947079](https://github.com/sgl-project/sglang/actions/runs/33780947079)
- **分支**: `gpt-xpu`
- **总耗时**: 36.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33780947079

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294178) |
| multimodal-gen-test-2-npu-a3 (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294201) |
| multimodal-gen-test-1-npu-a3 (1) | 35.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294265) |
| base-b-test-8-npu-a3 / run (0) | 35.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294278) |
| base-b-test-4-npu-a3 / run (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294292) |
| base-b-test-4-npu-a3 / run (1) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294294) |
| multimodal-gen-test-1-npu-a3 (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294303) |
| multimodal-gen-test-2-npu-a3 (1) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294335) |
| base-b-test-2-npu-a3 / run (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294352) |
| base-b-test-1-npu-a3 / run (0) | 35.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294434) |

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是 CI 基础设施问题，如日志上传延迟、文件被清理或路径配置错误，并非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294178

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、文件被清理或配置变更所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294201

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294265

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查CI配置中的存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294278

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294292

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294294

- **multimodal-gen-test-1-npu-a3 (0)**: 作业尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施或配置问题，与代码或模型性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294303

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖或缓存文件在 Azure Blob 中已被删除或路径错误，属于外部存储环境问题，需检查资源是否存在或更新下载链接。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294335

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或工件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294352

- **base-b-test-1-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失问题，需检查CI配置中的blob路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294434

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33780947079/job/100734294349) |


## [Run #33778427912](https://github.com/sgl-project/sglang/actions/runs/33778427912)
- **分支**: `fix/residency-calibration-probes`
- **总耗时**: 214.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33778427912

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (0) | 213.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043343) |
| multimodal-gen-test-1-npu-a3 (1) | 213.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043436) |
| multimodal-gen-test-2-npu-a3 (1) | 213.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043442) |
| multimodal-gen-test-1-npu-a3 (0) | 213.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，属于外部资源缺失。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043616) |

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob资源缺失，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043343

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明作业尝试下载的工件或数据文件在存储账户中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043436

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043442

- **multimodal-gen-test-1-npu-a3 (0)**: 作业失败并非代码或测试问题，而是下载或访问的Azure Blob文件（BlobNotFound）不存在，可能因资源被清理、路径错误或上传失败导致，需检查CI配置中的存储引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/33778427912/job/100726043616


## [Run #33775827167](https://github.com/sgl-project/sglang/actions/runs/33775827167)
- **分支**: `codex/diffusion-day0-variants`
- **总耗时**: 63.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33775827167

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775827167/job/100717337078) |
| multimodal-gen-test-2-npu-a3 (0) | 63.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775827167/job/100717337136) |
| multimodal-gen-test-2-npu-a3 (1) | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775827167/job/100717337166) |
| multimodal-gen-test-1-npu-a3 (0) | 63.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775827167/job/100717337180) |

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775827167/job/100717337078

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775827167/job/100717337136

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个存储对象缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775827167/job/100717337166

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775827167/job/100717337180


## [Run #33775312038](https://github.com/sgl-project/sglang/actions/runs/33775312038)
- **分支**: `main`
- **总耗时**: 8.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33775312038

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614333) |
| multimodal-gen-test-2-npu-a3 (0) | 7.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614388) |
| multimodal-gen-test-1-npu-a3 (0) | 7.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614401) |
| multimodal-gen-test-1-npu-a3 (1) | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614527) |
| base-b-test-16-npu-a3 / run (0) | 7.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614537) |
| base-b-test-1-npu-a3 / run (0) | 7.8min | 环境问题 | CI作业因Azure Blob存储中指定文件不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614595) |
| base-b-test-2-npu-a3 / run (0) | 7.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614604) |
| base-b-test-4-npu-a3 / run (0) | 7.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614624) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614713) |
| base-b-test-4-npu-a3 / run (1) | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614785) |
| base-a-test-1-npu-a2 / run (0) | 7.7min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614849) |

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失的环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614333

- **multimodal-gen-test-2-npu-a3 (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614388

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明作业依赖的某个文件或工件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614401

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614527

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614537

- **base-b-test-1-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业依赖的某个blob文件已被删除或路径错误，导致无法下载资源，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614595

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是因为日志文件被清理、路径错误或上传失败，属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614604

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614624

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614713

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614785

- **base-a-test-1-npu-a2 / run (0)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业中断，属于环境或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775312038/job/100715614849


## [Run #33775089892](https://github.com/sgl-project/sglang/actions/runs/33775089892)
- **分支**: `fix/residency-calibration-probes`
- **总耗时**: 33.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33775089892

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 32.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775089892/job/100714870417) |
| multimodal-gen-test-1-npu-a3 (0) | 32.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775089892/job/100714870438) |
| multimodal-gen-test-2-npu-a3 (0) | 32.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775089892/job/100714870460) |
| multimodal-gen-test-1-npu-a3 (1) | 32.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取必要数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33775089892/job/100714870474) |

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未正确生成，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775089892/job/100714870417

- **multimodal-gen-test-1-npu-a3 (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775089892/job/100714870438

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于基础设施或资源缺失问题，与代码或模型性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775089892/job/100714870460

- **multimodal-gen-test-1-npu-a3 (1)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是文件被清理、路径错误或上传失败，属于外部存储环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33775089892/job/100714870474


## [Run #33772533252](https://github.com/sgl-project/sglang/actions/runs/33772533252)
- **分支**: `fix/residency-calibration-probes`
- **总耗时**: 25.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33772533252

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 25.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，属于外部资源缺失。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33772533252/job/100706280759) |
| multimodal-gen-test-1-npu-a3 (1) | 25.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33772533252/job/100706280851) |
| multimodal-gen-test-2-npu-a3 (0) | 25.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33772533252/job/100706280874) |
| multimodal-gen-test-2-npu-a3 (1) | 25.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33772533252/job/100706281046) |

- **multimodal-gen-test-1-npu-a3 (0)**: 作业失败原因是下载或访问Azure Blob存储中的文件时返回BlobNotFound错误，可能是构建产物或依赖文件未上传或已被删除，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33772533252/job/100706280759

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33772533252/job/100706280851

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33772533252/job/100706280874

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是存储配置错误或上游任务未成功产出，需检查相关存储路径和依赖任务。
  链接: https://github.com/sgl-project/sglang/actions/runs/33772533252/job/100706281046


## [Run #33765202272](https://github.com/sgl-project/sglang/actions/runs/33765202272)
- **分支**: `main`
- **总耗时**: 104.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33765202272

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 103.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33765202272/job/100681349863) |
| multimodal-gen-test-2-npu-a3 (0) | 103.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33765202272/job/100681349972) |
| multimodal-gen-test-2-npu-a3 (1) | 103.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33765202272/job/100681349998) |
| multimodal-gen-test-1-npu-a3 (1) | 103.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33765202272/job/100681350044) |

- **multimodal-gen-test-1-npu-a3 (0)**: 作业失败原因是访问Azure Blob存储时返回BlobNotFound错误，即请求的资源不存在。这通常是由于日志或依赖文件被清理、路径错误或上传失败导致，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33765202272/job/100681349863

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的日志或工件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33765202272/job/100681349972

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33765202272/job/100681349998

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33765202272/job/100681350044


## [Run #33764826439](https://github.com/sgl-project/sglang/actions/runs/33764826439)
- **分支**: `mick/diffusion-auto-residency`
- **总耗时**: 105.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33764826439

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 105.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764826439/job/100680053485) |
| multimodal-gen-test-1-npu-a3 (0) | 105.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764826439/job/100680053583) |
| multimodal-gen-test-1-npu-a3 (1) | 105.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764826439/job/100680053644) |
| multimodal-gen-test-2-npu-a3 (0) | 105.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764826439/job/100680053648) |

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已被删除或路径错误，属于基础设施或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764826439/job/100680053485

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764826439/job/100680053583

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 blob 资源已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764826439/job/100680053644

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764826439/job/100680053648


## [Run #33764028079](https://github.com/sgl-project/sglang/actions/runs/33764028079)
- **分支**: `main`
- **总耗时**: 9.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33764028079

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 7.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388316) |
| base-b-test-1-npu-a3 / run (0) | 7.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388349) |
| base-b-test-4-npu-a3 / run (0) | 7.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388390) |
| base-b-test-4-npu-a3 / run (1) | 7.5min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388412) |
| base-b-test-2-npu-a3 / run (0) | 7.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388414) |
| base-a-test-1-npu-a2 / run (0) | 7.7min | 环境问题 | 容器执行失败，下载triton-ascend依赖时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388432) |
| base-b-test-16-npu-a3 / run (0) | 7.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388502) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388316

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388349

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388390

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是上游产物未上传或过期，属于基础设施/资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388412

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是上传失败、过期清理或配置问题，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388414

- **base-a-test-1-npu-a2 / run (0)**: 作业在安装triton-ascend==3.2.1.dev20260530时下载188.5MB的wheel包，下载过程中自定义容器实现执行失败，导致作业终止。属于NPU环境依赖下载或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388432

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/环境配置问题，需检查 blob 路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33764028079/job/100677388502


## [Run #33763934650](https://github.com/sgl-project/sglang/actions/runs/33763934650)
- **分支**: `ltx2`
- **总耗时**: 8.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33763934650

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 7.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980037) |
| multimodal-gen-test-2-npu-a3 (1) | 7.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980105) |
| multimodal-gen-test-2-npu-a3 (0) | 7.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980116) |
| multimodal-gen-test-1-npu-a3 (0) | 7.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980134) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980138) |
| base-b-test-16-npu-a3 / run (0) | 7.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980185) |
| base-b-test-2-npu-a3 / run (0) | 7.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980245) |
| base-b-test-1-npu-a3 / run (0) | 7.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980260) |
| base-b-test-4-npu-a3 / run (1) | 7.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980266) |
| base-a-test-1-npu-a2 / run (0) | 6.9min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980315) |
| base-b-test-4-npu-a3 / run (0) | 7.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980475) |

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储对象已被删除或路径错误，可能是CI依赖的模型权重或缓存文件缺失，需检查资源上传或路径配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980037

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是存储配置错误或上游任务未成功产出，需检查相关存储路径及依赖任务状态。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980105

- **multimodal-gen-test-2-npu-a3 (0)**: 作业在下载或访问某个blob资源时，返回BlobNotFound错误，可能是资源被删除、路径错误或存储账户配置问题，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980116

- **multimodal-gen-test-1-npu-a3 (0)**: 作业尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980134

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980138

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传失败、路径错误或存储被清理所致，属于基础设施或配置问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980185

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该文件不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980245

- **base-b-test-1-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980260

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980266

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在安装triton-ascend依赖后，runner报错“Executing the custom container implementation failed”，属于自托管runner环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980315

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33763934650/job/100676980475


## [Run #33761259703](https://github.com/sgl-project/sglang/actions/runs/33761259703)
- **分支**: `hybrid-spec-decoding`
- **总耗时**: 41.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33761259703

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 41.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089147) |
| multimodal-gen-test-2-npu-a3 (0) | 41.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089156) |
| multimodal-gen-test-1-npu-a3 (0) | 41.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089224) |
| multimodal-gen-test-2-npu-a3 (1) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089242) |
| base-b-test-8-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089426) |
| base-b-test-16-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089451) |
| base-b-test-2-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089490) |
| base-b-test-4-npu-a3 / run (1) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089625) |
| base-b-test-4-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089661) |
| base-b-test-1-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089813) |

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089147

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089156

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失或配置错误，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089224

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089242

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089426

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089451

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089490

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089625

- **base-b-test-4-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 对象，但该对象已被删除或路径错误，返回 BlobNotFound 错误。这通常是 CI 配置中引用了过期或未上传的日志文件所致，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089661

- **base-b-test-1-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 对象，但该对象已被删除或路径错误，返回 BlobNotFound 错误。这通常是 CI 配置中引用了过期或未上传的日志文件所致，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089813

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33761259703/job/100668089864) |


## [Run #33760632805](https://github.com/sgl-project/sglang/actions/runs/33760632805)
- **分支**: `codex/diffusion-day0-variants`
- **总耗时**: 157.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33760632805

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 157.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33760632805/job/100665911556) |
| multimodal-gen-test-2-npu-a3 (0) | 157.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33760632805/job/100665911628) |
| multimodal-gen-test-2-npu-a3 (1) | 157.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33760632805/job/100665911717) |
| multimodal-gen-test-1-npu-a3 (1) | 157.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33760632805/job/100665911797) |

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储对象已被删除或路径错误，可能是CI配置中引用的模型权重或测试数据未正确上传，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/33760632805/job/100665911556

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试下载的依赖文件或缓存已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33760632805/job/100665911628

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33760632805/job/100665911717

- **multimodal-gen-test-1-npu-a3 (1)**: 作业尝试下载或引用一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题，与代码或模型性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33760632805/job/100665911797


## [Run #33758314049](https://github.com/sgl-project/sglang/actions/runs/33758314049)
- **分支**: `feat/planner-unified-memory`
- **总耗时**: 60.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33758314049

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 59.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33758314049/job/100658245628) |
| multimodal-gen-test-1-npu-a3 (1) | 59.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33758314049/job/100658245667) |
| multimodal-gen-test-1-npu-a3 (0) | 59.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33758314049/job/100658245686) |
| multimodal-gen-test-2-npu-a3 (0) | 59.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33758314049/job/100658245725) |

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到，可能是日志或依赖文件被删除或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33758314049/job/100658245628

- **multimodal-gen-test-1-npu-a3 (1)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33758314049/job/100658245667

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33758314049/job/100658245686

- **multimodal-gen-test-2-npu-a3 (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33758314049/job/100658245725


---
*Auto-generated by npu_pr_monitor.py*