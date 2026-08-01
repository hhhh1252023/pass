# NPU CI 执行监控
**生成时间**: 2026-08-01 12:36 UTC
**分析 Run 数**: 37

---

## [Run #29411683974](https://github.com/sgl-project/sglang/actions/runs/29411683974)
- **分支**: `fix-kv-cache-aiter-memory-allocation`
- **总耗时**: 84.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29411683974

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073760) |
| stage-b-test-16-npu-a3 | 84.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073807) |
| multimodal-gen-test-1-npu-a3 | 84.1min | 环境问题 | 日志中引用的Azure Blob存储文件不存在，导致作业无法获取必要数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073818) |
| multimodal-gen-test-2-npu-a3 | 84.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073835) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340074158) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073760

- **stage-b-test-16-npu-a3**: 作业在下载或访问某个blob时失败，错误码BlobNotFound表明该资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073807

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问某个Blob时返回BlobNotFound错误，可能是文件被删除、路径错误或上传未完成，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073818

- **multimodal-gen-test-2-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件可能已被删除、路径错误或未上传成功，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073835

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340074158

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073794) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073812) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073824) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073893) |


## [Run #29404997476](https://github.com/sgl-project/sglang/actions/runs/29404997476)
- **分支**: `codex/kimi-vlm-warmup`
- **总耗时**: 245.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29404997476

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254336) |
| multimodal-gen-test-2-npu-a3 | 62.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254342) |
| stage-b-test-4-npu-a3 | 48.6min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254461) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254336

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误信息，仅有Node.js弃用警告和上传artifact时无文件提示，无法判断失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254342

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，5个NPU测试有4个通过，仅test_npu_llada2_mini.py失败（耗时895秒），其余测试正常，表明该测试用例本身存在代码或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254461

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254296) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254297) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254345) |
| stage-b-test-1-npu-a2 (1) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254357) |
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254376) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254851) |


## [Run #29403277122](https://github.com/sgl-project/sglang/actions/runs/29403277122)
- **分支**: `hicache-shm-allocator`
- **总耗时**: 255.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29403277122

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 46.7min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759004) |
| multimodal-gen-test-2-npu-a3 | 63.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759028) |
| multimodal-gen-test-1-npu-a3 | 62.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759044) |

- **stage-b-test-4-npu-a3**: 作业中5个NPU测试有4个通过，仅test_npu_llada2_mini.py失败（耗时855秒），其余测试正常，表明该测试用例本身存在代码或逻辑错误，非环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759004

- **multimodal-gen-test-2-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未包含多模态生成测试的具体执行步骤或错误输出，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759028

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败的具体错误，仅显示Node 20弃用警告和diffusion-failures目录无文件上传提示，无法判断真实失败原因，可能为测试未运行或结果未生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759044

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759012) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759040) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759052) |
| stage-b-test-1-npu-a2 (0) | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759062) |
| stage-b-test-1-npu-a2 (1) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759064) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759549) |


## [Run #29402588321](https://github.com/sgl-project/sglang/actions/runs/29402588321)
- **分支**: `bbuf/kernels-fill-noncuda-coverage`
- **总耗时**: 240.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29402588321

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 48.3min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，其余4个测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494369) |
| multimodal-gen-test-1-npu-a3 | 54.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494407) |

- **stage-b-test-4-npu-a3**: 测试test_npu_llada2_mini.py执行失败（exit code 1），耗时925秒，其余4个NPU测试均通过，表明该测试用例本身存在问题，可能是代码逻辑或环境依赖导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494369

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494407

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494370) |
| stage-b-test-1-npu-a2 (0) | 43.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494402) |
| multimodal-gen-test-2-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494419) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494426) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494536) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494580) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494818) |


## [Run #29397784488](https://github.com/sgl-project/sglang/actions/runs/29397784488)
- **分支**: `skip_tokenizer_multi_worker`
- **总耗时**: 108.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29397784488

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 108.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337067359) |
| stage-b-test-4-npu-a3 | 108.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337067381) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337067359

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337067381

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068068) |
| stage-b-test-16-npu-a3 | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068193) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068214) |
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068316) |
| multimodal-gen-test-2-npu-a3 | 35.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068352) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068415) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068466) |


## [Run #29392287525](https://github.com/sgl-project/sglang/actions/runs/29392287525)
- **分支**: `feat/sm120_glm51`
- **总耗时**: 245.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29392287525

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.6min | 其他 | 作业日志被截断，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314544633) |
| stage-b-test-4-npu-a3 | 33.1min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314544668) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示上传artifact时未找到diffusion-failures目录，说明测试可能未产生失败文件，但无法确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314544633

- **stage-b-test-4-npu-a3**: 测试套件中5个测试有2个通过，1个失败。失败的测试文件为test_npu_llada2_mini.py，执行耗时870秒，超过预估的400秒，最终返回退出码1，导致整个CI作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314544668

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314545698) |
| stage-b-test-1-npu-a2 (1) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314562008) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314567937) |
| multimodal-gen-test-2-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314571900) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314573550) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314573808) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314576986) |


## [Run #29350702274](https://github.com/sgl-project/sglang/actions/runs/29350702274)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 32.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29350702274

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 32.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965875) |
| multimodal-gen-test-1-npu-a3 | 32.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965896) |
| multimodal-gen-test-2-npu-a3 | 32.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965913) |
| stage-b-test-4-npu-a3 | 32.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965940) |
| stage-b-test-1-npu-a2 (0) | 31.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965949) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 32.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145968191) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965875

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965896

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965913

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965940

- **stage-b-test-1-npu-a2 (0)**: 日志显示sglang服务正常启动并处理请求，但随后出现'Executing the custom container implementation failed'错误，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965949

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145968191

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965961) |
| stage-b-test-2-npu-a2 (0) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965968) |
| stage-b-test-2-npu-a2 (1) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145966023) |


## [Run #29345341583](https://github.com/sgl-project/sglang/actions/runs/29345341583)
- **分支**: `remove-qserve-quantization`
- **总耗时**: 69.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29345341583

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 69.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527679) |
| multimodal-gen-test-2-npu-a3 | 69.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527774) |
| stage-b-test-4-npu-a3 | 69.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527786) |
| multimodal-gen-test-1-npu-a3 | 69.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527803) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 69.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127528392) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527679

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527774

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527786

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527803

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127528392

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527714) |
| stage-b-test-2-npu-a2 (1) | 26.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527726) |
| stage-b-test-1-npu-a2 (1) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527749) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527759) |


## [Run #29345312974](https://github.com/sgl-project/sglang/actions/runs/29345312974)
- **分支**: `brayden/fuse-trtllm-gen-prologue-kernels`
- **总耗时**: 77.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29345312974

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411261) |
| multimodal-gen-test-1-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411271) |
| stage-b-test-4-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411278) |
| multimodal-gen-test-2-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411298) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411739) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411261

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411271

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或存储配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411278

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411298

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或测试数据）在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411739

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411243) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411290) |
| stage-b-test-1-npu-a2 (0) | 50.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411341) |
| stage-b-test-1-npu-a2 (1) | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411357) |


## [Run #29344715585](https://github.com/sgl-project/sglang/actions/runs/29344715585)
- **分支**: `brayden/remove-aot-router-fused-a-gemm`
- **总耗时**: 77.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29344715585

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 76.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346204) |
| stage-b-test-4-npu-a3 | 76.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346225) |
| stage-b-test-16-npu-a3 | 76.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346237) |
| multimodal-gen-test-2-npu-a3 | 76.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346318) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 76.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346904) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346204

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346225

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346237

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被删除、路径错误或上传失败。这属于外部依赖环境问题，需检查CI配置中的存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346318

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346904

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346108) |
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346145) |
| stage-b-test-1-npu-a2 (1) | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346147) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346162) |


## [Run #29342296396](https://github.com/sgl-project/sglang/actions/runs/29342296396)
- **分支**: `bbuf/hpc-ops-attention-backend`
- **总耗时**: 9.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29342296396

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083181) |
| stage-b-test-16-npu-a3 | 8.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083214) |
| stage-b-test-2-npu-a2 (0) | 7.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083215) |
| multimodal-gen-test-2-npu-a3 | 8.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083242) |
| stage-b-test-1-npu-a2 (1) | 8.2min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083268) |
| stage-b-test-4-npu-a3 | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083282) |
| stage-b-test-2-npu-a2 (1) | 7.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083285) |
| stage-b-test-1-npu-a2 (0) | 7.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083390) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083854) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083181

- **stage-b-test-16-npu-a3**: 作业在下载或访问Azure Blob存储中的某个文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083214

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试在运行约5分钟后，出现"Executing the custom container implementation failed"错误，提示联系self-hosted runner管理员，属于NPU环境或容器基础设施问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083215

- **multimodal-gen-test-2-npu-a3**: 作业尝试下载或访问一个不存在的 Blob 文件（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083242

- **stage-b-test-1-npu-a2 (1)**: 作业在运行test_npu_graph_tp1_bf16.py时，自定义容器实现执行失败，导致测试中断。日志显示容器环境问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083268

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083282

- **stage-b-test-2-npu-a2 (1)**: 日志显示服务启动后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083285

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试在运行约20秒后，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083390

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或数据文件）在 Azure Blob 中缺失，可能是路径错误或文件未上传，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083854


## [Run #29340140589](https://github.com/sgl-project/sglang/actions/runs/29340140589)
- **分支**: `mmangkad/torch-2.12`
- **总耗时**: 178.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29340140589

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.1min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507629) |
| multimodal-gen-test-1-npu-a3 | 51.5min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507678) |
| stage-b-test-16-npu-a3 | 44.4min | 环境问题 | NPU Deepep测试失败，服务启动后无输出，可能因环境或配置问题导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507781) |

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，5个测试有4个通过，但test_npu_llada2_mini.py测试失败（退出码1），耗时874秒。该测试属于dllm模块，可能涉及LLaDA2模型相关功能，需检查该测试的具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507629

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507678

- **stage-b-test-16-npu-a3**: 测试test_npu_deepep.py启动DeepSeek-R1模型服务后无响应，2415秒后超时失败。可能因NPU资源不足、模型加载失败或Deepep配置不兼容导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507781

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507608) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507622) |
| multimodal-gen-test-2-npu-a3 | 33.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507696) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507704) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507720) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109508198) |


## [Run #29339745832](https://github.com/sgl-project/sglang/actions/runs/29339745832)
- **分支**: `feat/semantic-radix-backend`
- **总耗时**: 177.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29339745832

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 53.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169408) |
| stage-b-test-4-npu-a3 | 39.1min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，3/5通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169568) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误，仅有Node 20弃用警告和diffusion-failures目录无文件上传提示，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169408

- **stage-b-test-4-npu-a3**: 测试test_npu_llada2_mini.py执行失败（exit code 1），耗时888秒，其余3个测试通过。可能为代码逻辑错误或环境依赖问题，需查看具体测试日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169568

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169494) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169526) |
| stage-b-test-16-npu-a3 | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169575) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169577) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169588) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169593) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108170151) |


## [Run #29339195925](https://github.com/sgl-project/sglang/actions/runs/29339195925)
- **分支**: `htphan/fix-symm-mem-cuda-graph-deadlock`
- **总耗时**: 182.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29339195925

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 48.6min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358462) |
| multimodal-gen-test-1-npu-a3 | 56.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358612) |

- **stage-b-test-4-npu-a3**: 作业中5个NPU测试有4个通过，仅test_npu_llada2_mini.py失败（耗时906秒），属于该测试用例本身的代码或功能问题，非环境或超时导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358462

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时无文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358612

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 34.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358373) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358444) |
| stage-b-test-2-npu-a2 (1) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358459) |
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358464) |
| multimodal-gen-test-2-npu-a3 | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358475) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358514) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106359220) |


## [Run #29339049315](https://github.com/sgl-project/sglang/actions/runs/29339049315)
- **分支**: `idhanani/unified-radix-swa-fix`
- **总耗时**: 37.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29339049315

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 36.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760586) |
| stage-b-test-16-npu-a3 | 36.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760661) |
| multimodal-gen-test-2-npu-a3 | 36.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760672) |
| stage-b-test-4-npu-a3 | 36.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760750) |
| stage-b-test-1-npu-a2 (0) | 35.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760845) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 36.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105761298) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760586

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理或配置问题，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760661

- **multimodal-gen-test-2-npu-a3**: 作业日志中返回BlobNotFound错误，说明CI流程尝试访问的存储资源缺失或路径错误，属于环境配置或资源准备问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760672

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760750

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但在14:38:49出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760845

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105761298

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760656) |
| stage-b-test-2-npu-a2 (1) | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760668) |
| stage-b-test-1-npu-a2 (1) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760777) |


## [Run #29338838523](https://github.com/sgl-project/sglang/actions/runs/29338838523)
- **分支**: `bbuf/hpc-ops-attention-backend`
- **总耗时**: 36.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29338838523

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 35.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089452) |
| stage-b-test-4-npu-a3 | 35.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089491) |
| multimodal-gen-test-1-npu-a3 | 35.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089499) |
| stage-b-test-1-npu-a2 (0) | 35.0min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089514) |
| multimodal-gen-test-2-npu-a3 | 35.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089544) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105090117) |

- **stage-b-test-16-npu-a3**: 作业在尝试下载或访问某个blob时，返回BlobNotFound错误，说明该资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089452

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089491

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业依赖的某个blob（可能是模型权重或测试数据）已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089499

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到51%时出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089514

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089544

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105090117

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089494) |
| stage-b-test-1-npu-a2 (1) | 32.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089504) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089559) |


## [Run #29338486836](https://github.com/sgl-project/sglang/actions/runs/29338486836)
- **分支**: `pr_add_multi_stream_gemm_fusion`
- **总耗时**: 172.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29338486836

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.7min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，4/5通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825432) |
| multimodal-gen-test-1-npu-a3 | 52.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825457) |

- **stage-b-test-4-npu-a3**: 测试test_npu_llada2_mini.py返回退出码1，耗时915秒，其余4个测试均通过。该测试涉及dllm功能，可能是代码逻辑或环境配置问题导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825432

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅有Node.js版本弃用警告和上传diffusion-failures产物时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825457

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825389) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825413) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825429) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825461) |
| multimodal-gen-test-2-npu-a3 | 37.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825467) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825504) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825904) |


## [Run #29338080660](https://github.com/sgl-project/sglang/actions/runs/29338080660)
- **分支**: `kernels/phase25-vendored`
- **总耗时**: 172.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29338080660

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.6min | 代码错误 | 测试 test_npu_llada2_mini.py 失败，返回退出码 1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463204) |
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463285) |

- **stage-b-test-4-npu-a3**: 在 NPU A3 环境下，5 个测试中 4 个通过，仅 test_npu_llada2_mini.py 失败（耗时 870 秒），具体错误信息未在日志中显示，需进一步查看该测试的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463204

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，仅显示上传diffusion-failures目录时无文件，未包含测试执行细节或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463285

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463067) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463096) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463162) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463183) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463203) |
| multimodal-gen-test-2-npu-a3 | 51.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463260) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463737) |


## [Run #29337703742](https://github.com/sgl-project/sglang/actions/runs/29337703742)
- **分支**: `kernels/phase25-dsa-dsv4`
- **总耗时**: 176.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29337703742

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.0min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，其余4个测试通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137833) |
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137846) |

- **stage-b-test-4-npu-a3**: 测试test_npu_llada2_mini.py返回退出码1，耗时878秒，其余4个NPU测试均通过。该测试可能因代码逻辑错误或环境配置问题导致失败，需查看具体错误日志定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137833

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行结果或错误信息，仅有Node.js 20弃用警告和diffusion-failures目录无文件上传提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137846

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137829) |
| multimodal-gen-test-2-npu-a3 | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137844) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137850) |
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137859) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137874) |
| stage-b-test-16-npu-a3 | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137898) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101138266) |


## [Run #29337380180](https://github.com/sgl-project/sglang/actions/runs/29337380180)
- **分支**: `fuse-gate-gemv-into-append`
- **总耗时**: 153.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29337380180

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 52.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境警告和上传产物提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038638) |
| stage-b-test-4-npu-a3 | 47.0min | 代码错误 | 测试用例 test_npu_llada2_mini.py 执行失败，返回退出码 1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038673) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行结果或错误信息，仅有Node.js弃用警告和diffusion-failures目录无文件上传的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038638

- **stage-b-test-4-npu-a3**: 在 NPU A3 环境下，5 个测试中 4 个通过，仅 test_npu_llada2_mini.py 失败，耗时 878 秒，非超时问题，属于该测试用例本身的代码或功能错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038673

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 47.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038663) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038672) |
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038708) |
| stage-b-test-1-npu-a2 (1) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038724) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038745) |
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038763) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100039309) |


## [Run #29330274051](https://github.com/sgl-project/sglang/actions/runs/29330274051)
- **分支**: `feat/lora-merge-ipc-update`
- **总耗时**: 162.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29330274051

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 57.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29330274051/job/87076364690) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，仅显示上传diffusion-failures目录时无文件，未包含测试执行细节或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29330274051/job/87076364690

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29330274051/job/87076364647) |


## [Run #29329839778](https://github.com/sgl-project/sglang/actions/runs/29329839778)
- **分支**: `fix/glm-tool-parser-escapes`
- **总耗时**: 171.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29329839778

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.9min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，4/5通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930348) |
| multimodal-gen-test-1-npu-a3 | 62.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930350) |
| stage-b-test-2-npu-a2 (1) | 9.3min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930351) |
| stage-b-test-2-npu-a2 (0) | 4.0min | 环境问题 | pip下载依赖时网络连接中断，导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930378) |

- **stage-b-test-4-npu-a3**: 测试test_npu_llada2_mini.py执行失败（exit code 1），耗时903秒，其余4个测试均通过。具体失败原因需查看该测试的详细日志，可能是代码逻辑或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930348

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有checkout、upload-artifact等步骤，且upload-artifact提示无文件上传。可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930350

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行到17%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930351

- **stage-b-test-2-npu-a2 (0)**: 在安装Python依赖包时，pip从远程下载文件过程中出现IncompleteRead错误（已读取75MB，预期188MB），网络连接中断导致ProtocolError，最终作业以退出码1失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930378

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930302) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930323) |
| multimodal-gen-test-2-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930384) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930407) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930703) |


## [Run #29328209895](https://github.com/sgl-project/sglang/actions/runs/29328209895)
- **分支**: `amd_fix_deepseekv4_0714`
- **总耗时**: 167.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29328209895

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.7min | 其他 | 测试用例 test_npu_llada2_mini.py 失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103353299) |

- **stage-b-test-4-npu-a3**: 该作业中5个NPU测试有4个通过，仅 test_npu_llada2_mini.py 失败（退出码1），耗时884秒。日志未显示具体错误原因，可能为用例本身问题或环境相关，需进一步查看该用例详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103353299

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354020) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354095) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354195) |
| stage-b-test-1-npu-a2 (1) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354466) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354467) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354651) |


## [Run #29322097405](https://github.com/sgl-project/sglang/actions/runs/29322097405)
- **分支**: `main`
- **总耗时**: 139.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29322097405

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 138.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683822) |
| multimodal-gen-test-1-npu-a3 | 21.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683835) |
| multimodal-gen-test-2-npu-a3 | 138.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683851) |
| stage-b-test-2-npu-a2 (0) | 4.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683856) |
| stage-b-test-4-npu-a3 | 33.0min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683894) |
| stage-b-test-2-npu-a2 (1) | 3.5min | 环境问题 | pip下载依赖时网络连接中断，导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683933) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | 其他 | 日志显示测试状态为pass，但作业被标记为失败，可能是后续步骤或资源清理问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049684043) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683822

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683835

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683851

- **stage-b-test-2-npu-a2 (0)**: 作业在运行测试命令后，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683856

- **stage-b-test-4-npu-a3**: 测试套件中2/5通过，失败用例为test_npu_llada2_mini.py，耗时844秒超过预估400秒，返回退出码1，导致整个作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683894

- **stage-b-test-2-npu-a2 (1)**: 在安装Python依赖包时，pip从网络下载文件过程中连接中断（IncompleteRead），仅读取了约17MB数据，还有170MB未下载完成，导致安装失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683933

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中test_status为pass，plog备份正常，但作业最终失败。可能原因是后续步骤（如artifact上传或资源清理）出错，或日志被截断未显示真实错误。建议查看完整日志末尾的失败步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049684043

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683837) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683862) |


## [Run #29321466158](https://github.com/sgl-project/sglang/actions/runs/29321466158)
- **分支**: `kernels/phase25-linear`
- **总耗时**: 141.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29321466158

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 46.6min | 代码错误 | NPU测试test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612713) |
| multimodal-gen-test-1-npu-a3 | 52.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612720) |

- **stage-b-test-4-npu-a3**: 作业中5个NPU测试有4个通过，仅test_npu_llada2_mini.py失败（耗时855秒），属于该测试用例本身的代码或运行错误，非环境或超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612713

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体错误信息，仅显示上传artifact时未找到diffusion-failures目录，说明测试可能未产生失败文件，但实际失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612720

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612722) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612759) |
| stage-b-test-1-npu-a2 (0) | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612775) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612776) |
| multimodal-gen-test-2-npu-a3 | 35.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612787) |
| stage-b-test-2-npu-a2 (0) | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612795) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047613145) |


## [Run #29321089757](https://github.com/sgl-project/sglang/actions/runs/29321089757)
- **分支**: `codex/support-fal-ideogram-v4-fast-instant`
- **总耗时**: 140.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29321089757

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 53.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798879) |
| stage-b-test-4-npu-a3 | 47.6min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798898) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含Node 20弃用警告、上传diffusion-failures目录时无文件等常规信息，未出现测试执行或失败的具体错误，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798879

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，测试test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py失败（exit code 1），其余4个测试均通过。该测试用例本身存在代码或环境兼容性问题，导致整体作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798898

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798865) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798875) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798888) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798901) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798907) |
| stage-b-test-16-npu-a3 | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798922) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046799478) |


## [Run #29319625597](https://github.com/sgl-project/sglang/actions/runs/29319625597)
- **分支**: `cctry/kv-to-page-indices-on-device`
- **总耗时**: 151.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29319625597

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 33.6min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655243) |
| multimodal-gen-test-1-npu-a3 | 51.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655282) |

- **stage-b-test-4-npu-a3**: 作业中5个测试有3个通过，2个失败，其中test_npu_llada2_mini.py测试用例执行失败（exit code 1），导致整个作业以非零退出码结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655243

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、checkout、上传artifact等常规步骤，未出现测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655282

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655259) |
| multimodal-gen-test-2-npu-a3 | 35.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655270) |
| stage-b-test-16-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655295) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655300) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655301) |
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655314) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041656019) |


## [Run #29319427123](https://github.com/sgl-project/sglang/actions/runs/29319427123)
- **分支**: `pr_add_multi_stream_gemm_fusion`
- **总耗时**: 152.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29319427123

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 3.2min | 环境问题 | pip下载依赖时网络连接中断，导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021057) |
| multimodal-gen-test-1-npu-a3 | 58.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021059) |
| stage-b-test-4-npu-a3 | 46.9min | 代码错误 | NPU测试中test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021085) |

- **stage-b-test-1-npu-a2 (1)**: 在安装Python依赖过程中，pip从网络下载包时出现IncompleteRead错误，连接中断导致下载不完整，最终安装失败。属于网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021057

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、checkout、上传artifact等步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021059

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，测试test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py运行失败（exit code 1），其余4个测试均通过。该测试耗时886秒，可能涉及LLADA2模型在NPU上的功能或性能问题，需检查该测试的具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021085

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021080) |
| multimodal-gen-test-2-npu-a3 | 34.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021102) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021106) |
| stage-b-test-16-npu-a3 | 18.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021115) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021134) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021329) |


## [Run #29318774860](https://github.com/sgl-project/sglang/actions/runs/29318774860)
- **分支**: `glm52/mtp-split-1-topk1`
- **总耗时**: 174.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29318774860

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.6min | 其他 | 测试用例 test_npu_llada2_mini.py 失败，其余4个用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935038) |
| multimodal-gen-test-1-npu-a3 | 62.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935048) |

- **stage-b-test-4-npu-a3**: 该作业中5个NPU测试用例有4个通过，仅 test_npu_llada2_mini.py 失败（退出码1），耗时868秒。日志未显示具体错误原因，可能是该用例本身存在问题或环境相关故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935038

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935048

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935095) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935195) |
| stage-b-test-1-npu-a2 (1) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935867) |
| multimodal-gen-test-2-npu-a3 | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935874) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935904) |
| stage-b-test-16-npu-a3 | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100936291) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100936568) |


## [Run #29289186291](https://github.com/sgl-project/sglang/actions/runs/29289186291)
- **分支**: `glm5/moe-output-output`
- **总耗时**: 249.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29289186291

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 33.4min | 超时 | 测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585300) |
| multimodal-gen-test-1-npu-a3 | 62.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585316) |

- **stage-b-test-4-npu-a3**: test_npu_llada2_mini.py 运行超过900秒（预计400秒），超时被强制终止，返回退出码1，导致作业失败。其余4个测试中2个通过，2个未运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585300

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行步骤或错误信息，仅显示Node.js 20弃用警告和上传artifact时无文件。可能因日志截断或作业在测试前被取消，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585316

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585994) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586091) |
| stage-b-test-1-npu-a2 (1) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586296) |
| multimodal-gen-test-2-npu-a3 | 45.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586354) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586460) |
| stage-b-test-16-npu-a3 | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586527) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586917) |


## [Run #29202367445](https://github.com/sgl-project/sglang/actions/runs/29202367445)
- **分支**: `dp-attn-free-port-block`
- **总耗时**: 60.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29202367445

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 35.4min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819978) |
| multimodal-gen-test-1-npu-a3 | 58.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819982) |

- **stage-b-test-4-npu-a3**: 测试套件中test_npu_llada2_mini.py运行失败（exit code 1），其余4个测试均通过。该测试属于dllm功能模块，可能涉及代码逻辑或环境依赖问题，需查看具体错误日志定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819978

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819982

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819969) |
| multimodal-gen-test-2-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819973) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819981) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819983) |
| stage-b-test-2-npu-a2 (0) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819984) |
| stage-b-test-1-npu-a2 (0) | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819985) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675820186) |


## [Run #29201454263](https://github.com/sgl-project/sglang/actions/runs/29201454263)
- **分支**: `jit-dtype-trait-reduce-fix`
- **总耗时**: 26.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29201454263

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 25.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678065954) |

- **stage-b-test-4-npu-a3**: 日志显示测试运行到82%时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678065954

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066155) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066284) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066297) |
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066302) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066339) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066498) |


## [Run #29201063345](https://github.com/sgl-project/sglang/actions/runs/29201063345)
- **分支**: `jit_dsv4_c128_opt`
- **总耗时**: 42.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29201063345

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 36.1min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442393) |

- **stage-b-test-4-npu-a3**: 测试套件中test_npu_llada2_mini.py测试失败（exit code 1），其余4个测试均通过。该测试属于dllm功能模块，可能涉及LLaDA2模型相关代码问题，需检查该测试的具体报错日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442393

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442407) |
| stage-b-test-16-npu-a3 | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442412) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442417) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442440) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442463) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442620) |


## [Run #29199118191](https://github.com/sgl-project/sglang/actions/runs/29199118191)
- **分支**: `tom_refactor_202605a/primary/nonmech_model_runner`
- **总耗时**: 12.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29199118191

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 3.7min | 代码错误 | NPU W4A4量化测试失败，测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336072) |
| stage-b-test-2-npu-a2 (0) | 5.3min | 环境问题 | NPU测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336076) |
| stage-b-test-16-npu-a3 | 4.1min | 代码错误 | NPU DeepEP测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336080) |
| stage-b-test-1-npu-a2 (1) | 5.1min | 代码错误 | NPU采样后端测试失败，测试文件返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336081) |
| stage-b-test-1-npu-a2 (0) | 5.1min | 环境问题 | NPU测试用例test_npu_hicache_mha.py执行失败，退出码1，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336086) |
| multimodal-gen-test-2-npu-a3 | 7.3min | 其他 | 作业未发现明确失败原因，日志显示正常执行并上传产物。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336088) |
| stage-b-test-2-npu-a2 (1) | 5.0min | 代码错误 | NPU MLA FIA W8A8 INT8 测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336090) |
| multimodal-gen-test-1-npu-a3 | 11.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336103) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.8min | 环境问题 | 作业在启动阶段即被终止，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336237) |

- **stage-b-test-4-npu-a3**: test_npu_w4a4_quantization.py测试失败，0/5通过，退出码1。可能是量化实现或测试用例本身存在错误，需检查具体断言失败信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336072

- **stage-b-test-2-npu-a2 (0)**: 测试文件test_npu_graph_tp2_bf16.py执行失败，0/2测试通过，耗时74秒。日志未显示具体错误原因，可能是NPU环境配置或资源问题导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336076

- **stage-b-test-16-npu-a3**: 测试文件test_npu_deepep.py在运行55.59秒后失败，退出码为1，导致整个作业以255退出。具体失败原因需查看该测试的详细输出，可能是代码逻辑错误或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336080

- **stage-b-test-1-npu-a2 (1)**: test_npu_sampling_backend.py测试失败，0/4通过，耗时75秒。可能是采样后端实现或测试用例本身存在代码问题，需查看具体断言失败信息定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336081

- **stage-b-test-1-npu-a2 (0)**: 测试文件test/registered/ascend/basic_function/HiCache/test_npu_hicache_mha.py运行失败，返回退出码1，测试摘要显示0/5通过。可能是NPU环境配置问题或测试用例本身错误，需进一步查看详细日志定位具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336086

- **multimodal-gen-test-2-npu-a3**: 日志中仅包含Node.js 20弃用警告和未找到diffusion-failures目录的提示，未出现测试失败或错误信息，可能为作业提前结束或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336088

- **stage-b-test-2-npu-a2 (1)**: 测试文件 test_npu_mla_fia_w8a8int8.py 执行失败，返回退出码 1，测试摘要显示 0/2 通过，可能是功能实现或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336090

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行结果或错误信息，仅显示Node 20弃用警告和diffusion-failures目录无文件上传提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336103

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试用例，可能因基础设施或调度问题导致作业被提前取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336237


## [Run #29197701797](https://github.com/sgl-project/sglang/actions/runs/29197701797)
- **分支**: `tom_refactor_202605a/primary/nonmech_model_runner`
- **总耗时**: 44.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29197701797

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 4.3min | 代码错误 | NPU DeepEP 测试用例执行失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587419) |
| multimodal-gen-test-2-npu-a3 | 42.9min | 其他 | 作业失败但日志未显示明确错误，仅上传失败产物时提示无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587420) |
| stage-b-test-4-npu-a3 | 3.7min | 代码错误 | NPU MLA W8A8INT8 测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587431) |
| multimodal-gen-test-1-npu-a3 | 42.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587433) |
| stage-b-test-2-npu-a2 (0) | 5.2min | 代码错误 | NPU图模式TP2 BF16测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587438) |
| stage-b-test-1-npu-a2 (1) | 5.2min | 代码错误 | NPU采样后端测试失败，测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587440) |
| stage-b-test-2-npu-a2 (1) | 5.2min | 代码错误 | NPU测试用例test_npu_mla_fia_w8a8int8.py执行失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587450) |
| stage-b-test-1-npu-a2 (0) | 5.3min | 代码错误 | HiCache MHA测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587458) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.8min | 其他 | 日志被截断，未显示实际测试失败原因，仅包含作业初始化和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587730) |

- **stage-b-test-16-npu-a3**: 测试文件 test_npu_deepep.py 在 expert_parallelism 策略下运行失败，耗时54.56秒，0/1通过。具体错误信息未在日志中显示，但可判断为测试代码或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587419

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures目录时无文件，可能测试未生成失败产物或测试本身未执行成功，但缺少具体失败原因，需查看更详细日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587420

- **stage-b-test-4-npu-a3**: test_npu_mla_w8a8int8.py 测试执行失败，返回退出码1，导致整个作业失败。可能是测试用例本身存在代码问题或环境配置不兼容。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587431

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587433

- **stage-b-test-2-npu-a2 (0)**: 测试文件test_npu_graph_tp2_bf16.py执行失败，0/2测试通过，耗时74秒。可能是NPU图模式相关代码存在bug或环境配置问题，需查看具体测试输出定位错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587438

- **stage-b-test-1-npu-a2 (1)**: test_npu_sampling_backend.py测试失败，0/4通过，退出码1。可能因代码改动导致NPU采样后端功能异常，需检查相关实现。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587440

- **stage-b-test-2-npu-a2 (1)**: 测试文件test/registered/ascend/basic_function/runtime_opts/test_npu_mla_fia_w8a8int8.py在运行约74秒后失败，退出码为1，导致整个作业失败。具体错误信息未在日志中显示，但可判断为测试用例本身执行出错。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587450

- **stage-b-test-1-npu-a2 (0)**: test_npu_hicache_mha.py测试执行失败，退出码为1，导致整个作业失败。具体错误信息未在日志中显示，但测试文件本身存在问题或环境不满足测试要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587458

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志片段仅包含GitHub Actions运行器初始化、下载actions、设置环境及作业结束清理步骤，未包含测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587730


## [Run #29193506426](https://github.com/sgl-project/sglang/actions/runs/29193506426)
- **分支**: `tom_refactor_202605a/primary/nonmech_model_runner`
- **总耗时**: 56.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29193506426

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 4.2min | 代码错误 | NPU DeepEP 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277069) |
| multimodal-gen-test-1-npu-a3 | 55.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277076) |
| stage-b-test-2-npu-a2 (1) | 5.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277084) |
| stage-b-test-4-npu-a3 | 3.8min | 日志下载失败 | HTTPSConnectionPool(host='productionresultssa0.blob.core.windows.net', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277089) |
| stage-b-test-1-npu-a2 (1) | 5.2min | 代码错误 | NPU采样后端测试失败，测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277093) |
| stage-b-test-1-npu-a2 (0) | 5.3min | 代码错误 | HiCache MHA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277105) |
| stage-b-test-2-npu-a2 (0) | 5.1min | 代码错误 | NPU图测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277129) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.7min | 其他 | 日志被截断，无法看到实际测试结果，仅显示作业启动和清理过程。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277252) |

- **stage-b-test-16-npu-a3**: test_npu_deepep.py 测试在 expert_parallelism 场景下执行失败，耗时55.75秒，0/1通过，具体错误信息未在日志中显示，需查看测试详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277069

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时无文件，无法判断失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277076

- **stage-b-test-2-npu-a2 (1)**: 作业在启动NPU测试时，自定义容器实现执行失败，提示请联系自托管runner管理员。日志显示模型加载和tokenizer初始化正常，但容器运行环境出现问题，属于基础设施环境故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277084


- **stage-b-test-1-npu-a2 (1)**: test_npu_sampling_backend.py测试失败，0/4通过，耗时74秒。可能是采样逻辑或NPU后端实现有误，需查看具体断言或错误日志定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277093

- **stage-b-test-1-npu-a2 (0)**: test_npu_hicache_mha.py 测试执行失败，退出码为1，导致整个作业失败。具体错误信息在日志中被省略，需查看完整测试输出以确定根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277105

- **stage-b-test-2-npu-a2 (0)**: 测试文件test_npu_graph_tp2_bf16.py在NPU A2环境下运行失败，0/2测试通过，耗时74秒。具体错误信息未在日志中显示，可能是测试断言失败或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277129

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分被省略，只包含作业初始化、环境准备和结束清理信息，未显示测试执行的具体输出或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277252

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 37.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277133) |


## [Run #29193175682](https://github.com/sgl-project/sglang/actions/runs/29193175682)
- **分支**: `main`
- **总耗时**: 10.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29193175682

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 9.3min | 环境问题 | NPU作业在加载模型权重时出现Scheduler watchdog超时，导致容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366892) |
| stage-b-test-4-npu-a3 | 9.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366895) |
| stage-b-test-1-npu-a2 (1) | 9.2min | 环境问题 | 自定义容器执行失败，NPU后端算子回退到CPU导致性能问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366903) |
| multimodal-gen-test-1-npu-a3 | 9.2min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366910) |
| multimodal-gen-test-2-npu-a3 | 9.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366911) |
| stage-b-test-2-npu-a2 (0) | 9.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366918) |
| stage-b-test-1-npu-a2 (0) | 9.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366930) |
| stage-b-test-2-npu-a2 (1) | 9.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366936) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651367037) |

- **stage-b-test-16-npu-a3**: 日志显示在加载MoE模型权重（约80%进度）时，TP2 EP2进程触发Scheduler watchdog timeout（300秒），随后自定义容器执行失败。可能是NPU资源竞争或权重加载过慢导致看门狗超时，属于环境或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366892

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常，但在12:54:54时出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366895

- **stage-b-test-1-npu-a2 (1)**: 日志显示NPU后端不支持aten::_assert_async算子，回退到CPU执行，可能引发性能问题。随后出现自定义容器执行失败错误，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366903

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node 20弃用警告和上传artifact时无文件提示。可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366910

- **multimodal-gen-test-2-npu-a3**: 日志截断于中间省略部分，仅显示checkout、upload-artifact等步骤，无测试执行或失败详情，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366911

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行中（进度29%）时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是容器崩溃、资源限制或runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366918

- **stage-b-test-1-npu-a2 (0)**: 测试运行到58%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366930

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常（吞吐量正常），但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366936

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中未包含测试执行的关键输出，无法判断失败原因。可能因日志截断或作业在测试前被取消，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651367037


---
*Auto-generated by npu_pr_monitor.py*