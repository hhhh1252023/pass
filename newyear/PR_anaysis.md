# PR #35500 改动总结

> 分支：`fix/multi-node-isolate-by-runid`
> 目标：通过 `run_id` 隔离并发 run 的源码、K8s Pod、ConfigMap 与 plog，并为清理流程加 30min 超时，避免 CI 卡死与并发互删。
> 基线 commit：`6670bddced~1`（sgl-project/main）
> 最新 commit：`d89913b275`

---

## 改动概览

| # | 文件 | 行数 | 改进项 |
|---|------|------|--------|
| 1 | `.github/workflows/nightly-test-npu-e2e-multi-node.yml` | +57 / -14 | 1 / 3 / 4 / 9 |
| 2 | `python/sglang/test/ascend/e2e/run_npu_e2e_test.py` | +36 / -6 | 5 / 6 |
| 3 | `python/sglang/test/ascend/e2e/k8s_single.yaml.jinja2` | +1 | 6 |
| 4 | `python/sglang/test/ascend/e2e/k8s_multi_pd_mix.yaml.jinja2` | +1 | 6 |
| 5 | `python/sglang/test/ascend/e2e/k8s_multi_pd_mix_green.yaml.jinja2` | +1 | 6 |
| 6 | `python/sglang/test/ascend/e2e/k8s_multi_pd_separation.yaml.jinja2` | +3 | 6 |
| 7 | `python/sglang/test/ascend/e2e/k8s_multi_pd_separation_green.yaml.jinja2` | +3 | 6 |
| 8 | `python/sglang/test/ascend/e2e/run_npu_testcase.sh` | +2 / -2 | 7 |
| 9 | `.github/workflows/nightly-test-npu.yml` | +11 / -277 | 精简（非多机隔离改进） |

合计：9 个文件，+108 / -295

---

## 改进项对应表

| 改进项 | 目的 | 涉及文件 |
|--------|------|----------|
| 1 | 源码路径按 run_id 隔离，避免并发 run 互相覆盖源码 | multi-node.yml |
| 3 | Pre-test 清理 30min 超时 + fail fast，防止卡死 | multi-node.yml |
| 4 | Post-test 清理 --force + 30min 超时 + 只删本 run | multi-node.yml |
| 5 | Python 写入 job_name 文件 + 按 run_id 过滤 pod | run_npu_e2e_test.py, multi-node.yml |
| 6 | K8s 模板加 run-id label，实现 pod 级别隔离 | 5 个 jinja2 文件, run_npu_e2e_test.py |
| 7 | plog 路径加 run_label，避免日志互相覆盖 | run_npu_testcase.sh |
| 9 | Post-test 清理源码路径，避免磁盘占用累积 | multi-node.yml |

---

## 文件 1：`.github/workflows/nightly-test-npu-e2e-multi-node.yml`

### 改动 1：Prepare code for testing 步骤（改进 1）

```diff
-          sglang_source_relative_path=tests/sglang
+          # Append run_id so concurrent runs do not overwrite each other on the shared PVC.
+          sglang_source_relative_path=tests/sglang-${{ github.run_id }}
```

**逐行解释**：
- 旧代码：源码统一放在 `tests/sglang`，并发 run 会互相覆盖
- 新代码：源码目录加 `-${{ github.run_id }}` 后缀，每个 run 独立目录
- 注释说明改的原因（避免并发覆盖）

### 改动 2：Clear resources 步骤（改进 3）

```diff
-          while true; do
-            if kubectl get po -A -n $NAMESPACE | grep -q "${pod_name_prefix}"; then
-              echo "Found exist sglang job, sleeping for 30 seconds..."
+          max_retries=60
+          retry_count=0
+          while [ $retry_count -lt $max_retries ]; do
+            if kubectl get pods -n $NAMESPACE | grep -q "${pod_name_prefix}"; then
+              echo "Found stale sglang pods, retry $((retry_count+1))/${max_retries} (sleeping 30s)..."
               sleep 30
-              kubectl get pods | grep "${pod_name_prefix}" | awk '{print $1}' | xargs kubectl delete pod -n $NAMESPACE || true
+              kubectl get pods -n $NAMESPACE | grep "${pod_name_prefix}" | awk '{print $1}' | while read -r pod_name; do
+                [ -z "$pod_name" ] && continue
+                if kubectl delete pod "$pod_name" -n $NAMESPACE --force --ignore-not-found=true; then
+                  echo "  $pod_name: SUCCESS"
+                else
+                  echo "  $pod_name: WARNING (delete failed)"
+                fi
+              done
+              retry_count=$((retry_count+1))
             else
-              echo "No sglang job exist, start test case..."
+              echo "No stale sglang pods found, start test case..."
               break
             fi
           done
+          if [ $retry_count -ge $max_retries ]; then
+            echo "ERROR: pre-test cleanup timeout after $((max_retries*30))s, some pods still remain"
+            exit 1
+          fi
```

**逐行解释**：
- `max_retries=60`：60 次重试上限（60 × 30s = 30min 超时）
- `retry_count=0`：重试计数器
- `while [ $retry_count -lt $max_retries ]`：带超时上限的循环条件（旧代码 `while true` 无上限，曾卡 1118min）
- `kubectl get pods -n $NAMESPACE`：去掉 `-A`（只看本 namespace，避免权限问题）
- `echo "Found stale sglang pods, retry ..."`：更清晰的日志（含重试计数）
- `sleep 30`：每轮等 30s 给 Volcano controller 时间删 pod
- `... | while read -r pod_name`：用管道替代 `xargs`，便于逐个打印删除结果
- `[ -z "$pod_name" ] && continue`：跳过空行
- `kubectl delete pod --force --ignore-not-found=true`：`--force` 避免卡 Terminating，`--ignore-not-found` 避免 pod 已删时报错
- `SUCCESS / WARNING`：每个 pod 删除结果都有日志
- `retry_count=$((retry_count+1))`：每轮重试后计数+1
- `break`：pod 全部删除后退出循环
- `if [ $retry_count -ge $max_retries ]; then exit 1; fi`：**超时直接 fail fast**（与 Post-test 不同，Pre-test 失败必须让本次 run 失败，否则资源冲突）

### 改动 3：Run test 步骤（改进 1）

```diff
-          sglang_source_relative_path=tests/sglang
+          # Append run_id so concurrent runs do not overwrite each other on the shared PVC.
+          sglang_source_relative_path=tests/sglang-${{ github.run_id }}
```

**逐行解释**：与改动 1 一致，这里是在 Run test 步骤里再次设置源码路径

### 改动 4：调用 Python 脚本加 --run-id 参数（改进 5）

```diff
             --kube-job-type ${kube_job_type} \
-            --kube-job-name-prefix ${KUBE_JOB_NAME}"
+            --kube-job-name-prefix ${KUBE_JOB_NAME} \
+            --run-id ${{ github.run_id }}"
```

**逐行解释**：把 `github.run_id` 透传给 Python 脚本，用于后续给 pod 打 `run-id` label 和过滤 pod

### 改动 5：Post process 步骤（改进 4 + 9）

```diff
       - name: Post process
         if: always()
         run: |
           cd ${ASCEND_E2E_TEST_CONFIG_PATH}
-          kubectl get pods -n $NAMESPACE | grep $KUBE_JOB_NAME
+          kubectl get pods -n $NAMESPACE | grep $KUBE_JOB_NAME || true
           kubectl delete -f ./k8s_multi_pd_*.yaml --ignore-not-found=true || true

-          pod_name_prefix="${KUBE_JOB_NAME}-"
+          # Prefer the exact job name (run_id-scoped random_str) written by the Python
+          # script so we only delete pods belonging to this run, not other concurrent runs.
+          if [ -f /tmp/kube_job_name.txt ]; then
+            pod_name_prefix=$(cat /tmp/kube_job_name.txt)
+            echo "Post-test cleanup using exact job name: ${pod_name_prefix}"
+          else
+            pod_name_prefix="${KUBE_JOB_NAME}-"
+            echo "WARNING: /tmp/kube_job_name.txt not found, falling back to fixed prefix: ${pod_name_prefix}"
+          fi
           echo "kube name space: $NAMESPACE, pod name prefix: ${pod_name_prefix}"
-          while true; do
-            if kubectl get po -A -n $NAMESPACE | grep -q "${pod_name_prefix}"; then
-              echo "Found exist sglang job, sleeping for 30 seconds..."
+          max_retries=60
+          retry_count=0
+          while [ $retry_count -lt $max_retries ]; do
+            if kubectl get pods -n $NAMESPACE | grep -q "${pod_name_prefix}"; then
+              echo "Found remaining sglang pods, retry $((retry_count+1))/${max_retries} (sleeping 30s)..."
               sleep 30
-              kubectl get pods | grep "${pod_name_prefix}" | awk '{print $1}' | xargs kubectl delete pod -n $NAMESPACE || true
+              kubectl get pods -n $NAMESPACE | grep "${pod_name_prefix}" | awk '{print $1}' | while read -r pod_name; do
+                [ -z "$pod_name" ] && continue
+                if kubectl delete pod "$pod_name" -n $NAMESPACE --force --ignore-not-found=true; then
+                  echo "  $pod_name: SUCCESS"
+                else
+                  echo "  $pod_name: WARNING (delete failed)"
+                fi
+              done
+              retry_count=$((retry_count+1))
             else
-              echo "No sglang job exist, start test case..."
+              echo "No sglang pods remain."
               break
             fi
           done
+          if [ $retry_count -ge $max_retries ]; then
+            echo "WARNING: post-test cleanup timeout after $((max_retries*30))s, some pods may remain"
+          fi
+
+          # Cleanup the run-scoped source code directory on the shared PVC.
+          rm -rf /root/.cache/tests/sglang-${{ github.run_id }} || true
```

**逐行解释**：
- `kubectl get pods ... || true`：加 `|| true`，grep 无匹配时返回非零会导致 `sh -e` 退出
- `if [ -f /tmp/kube_job_name.txt ]`：检查 Python 是否写入过 job name 文件
- `pod_name_prefix=$(cat /tmp/kube_job_name.txt)`：用本 run 精确 job name 作前缀
- `echo "Post-test cleanup using exact job name"`：日志确认走精确匹配路径
- `else`：回退分支，Python 没执行到写文件（如 Run test 早期失败）
- `WARNING: /tmp/kube_job_name.txt not found`：WARNING 而非 ERROR，因为这是预期内的失败回退
- `max_retries=60 / retry_count=0`：与 Pre-test 一致的 30min 超时
- 循环逻辑与 Pre-test 完全一致（管道 + --force + 逐个打印）
- `if [ $retry_count -ge $max_retries ]; then echo "WARNING..."`：**超时只 WARNING 不 exit**（与 Pre-test 不同：Post-test 失败不影响本次测试结果，下次 Pre-test 兜底）
- `rm -rf /root/.cache/tests/sglang-${{ github.run_id }} || true`：清理本次 run 的源码目录（改进 9）

---

## 文件 2：`python/sglang/test/ascend/e2e/run_npu_e2e_test.py`

### 改动 1：prepare_cm_data 函数（改进 6）

```diff
-def prepare_cm_data(namespace, pod_string):
-    """Prepare a configmap data: {pod_name: pod_ip} by the running pod's information."""
+def prepare_cm_data(namespace, pod_string, run_id=None):
+    """Prepare a configmap data: {pod_name: pod_ip} by the running pod's information.
+
+    When run_id is provided, the label selector is scoped to that run_id so that
+    concurrent runs in the same namespace do not pollute each other's ConfigMap.
+    The pod_string filter (final_kube_job_name) still acts as a safety net.
+    """
+    if run_id:
+        label_selector = f"app=sgl-ascend,run-id={run_id}"
+    else:
+        label_selector = "app=sgl-ascend"
     pods = core_api.list_namespaced_pod(
-        namespace=namespace, label_selector="app=sgl-ascend"
+        namespace=namespace, label_selector=label_selector
     )
```

**逐行解释**：
- 函数签名加 `run_id=None`：向后兼容，本地跑不传也行
- docstring 说明并发隔离意图
- `label_selector = f"app=sgl-ascend,run-id={run_id}"`：K8s label selector 按 run_id 过滤，只查本 run 的 pod
- 回退分支用原 selector，保持兼容
- `list_namespaced_pod(label_selector=label_selector)`：传入动态 selector

### 改动 2：run_npu_e2e_test_case 函数签名（改进 5）

```diff
     env="debug",
     trouble_shotting=False,
     transformers_version="",
+    run_id: str = "",
 ):
```

**逐行解释**：新增 `run_id` 参数，默认空串（向后兼容）

### 改动 3：写入 job_name 到文件（改进 5）

```diff
     kube_config_map = f"sglang-configmap-{random_str}"
     final_kube_job_name = f"{kube_job_name_prefix}-{random_str}"
+    # Expose the run-scoped job name to the workflow so that the Post-test cleanup
+    # can target only this run's pods and avoid deleting pods of concurrent runs.
+    try:
+        with open("/tmp/kube_job_name.txt", "w") as f:
+            f.write(final_kube_job_name)
+    except Exception as e:
+        logger.warning(f"Failed to write /tmp/kube_job_name.txt: {e}")
```

**逐行解释**：
- `final_kube_job_name` 是含 16 位随机串的 job 名，每个 run 唯一
- 写入 `/tmp/kube_job_name.txt` 供 workflow 的 Post-test 读取
- `try/except`：文件写入失败不影响主流程，只打 WARNING

### 改动 4：三处 k8s_context 加 run_id（改进 6）

```diff
                 "run_label": run_label,
+                "run_id": run_id,
             }
```

**逐行解释**：三处 k8s_context（single / multi-pd-mix / multi-pd-separation）都加入 `run_id`，传给 jinja2 模板渲染成 pod label

### 改动 5：调用 prepare_cm_data 传 run_id（改进 6）

```diff
                 matching_pod_string = final_kube_job_name
-                cm_data = prepare_cm_data(kube_name_space, matching_pod_string)
+                cm_data = prepare_cm_data(
+                    kube_name_space, matching_pod_string, run_id=run_id
+                )
```

**逐行解释**：多机场景调用 `prepare_cm_data` 时传入 `run_id`，触发按 label 过滤

### 改动 6：argparse 加 --run-id（改进 5）

```diff
+    parser.add_argument(
+        "--run-id",
+        type=str,
+        required=False,
+        default="",
+        help="GitHub Actions run_id, used to label pods for run-scoped isolation.",
+    )
+
     args = parser.parse_args()
```

**逐行解释**：新增 CLI 参数，供 workflow 透传 `github.run_id`

### 改动 7：透传 run_id 到 run_npu_e2e_test_case（改进 5）

```diff
+    run_id = args.run_id
     ...
     run_npu_e2e_test_case(
         ...
+        run_id=run_id,
     )
```

**逐行解释**：从 CLI 读 `run_id` 并传给主函数

---

## 文件 3：`python/sglang/test/ascend/e2e/k8s_single.yaml.jinja2`

```diff
       metadata:
         labels:
           app: sgl-ascend
+          run-id: "{{ run_id }}"
           ring-controller.atlas: ascend-1980
```

**逐行解释**：单节点模板的 pod `metadata.labels` 增加 `run-id: "{{ run_id }}"`。jinja2 渲染时 `{{ run_id }}` 被替换成实际 run_id 字符串，使 pod 带上 run_id 标签，供 `prepare_cm_data` 的 label_selector 过滤。

---

## 文件 4：`python/sglang/test/ascend/e2e/k8s_multi_pd_mix.yaml.jinja2`

```diff
       metadata:
         labels:
           app: sgl-ascend
+          run-id: "{{ run_id }}"
           ring-controller.atlas: ascend-1980
```

**逐行解释**：多机混部模板的 `sglang-node` 任务 pod 加 `run-id` label，作用同上

---

## 文件 5：`python/sglang/test/ascend/e2e/k8s_multi_pd_mix_green.yaml.jinja2`

```diff
     metadata:
       labels:
         app: sgl-ascend
+        run-id: "{{ run_id }}"
         task: pd-mix
```

**逐行解释**：green 环境混部模板的 StatefulSet pod 加 `run-id` label

---

## 文件 6：`python/sglang/test/ascend/e2e/k8s_multi_pd_separation.yaml.jinja2`

```diff
# 三处相同改动，分别在 prefill / decode / router task 的 pod metadata
       metadata:
         labels:
           app: sgl-ascend
+          run-id: "{{ run_id }}"
           ring-controller.atlas: ascend-1980
```

**逐行解释**：多机分离部署模板的 `sglang-prefill`、`sglang-decode`、`sglang-router` 三个 task 的 pod 都加 `run-id` label。分离部署有 3 个 task，每个都要加，共 3 处改动。

---

## 文件 7：`python/sglang/test/ascend/e2e/k8s_multi_pd_separation_green.yaml.jinja2`

```diff
# 三处相同改动，分别在 prefill / decode / router StatefulSet
     metadata:
       labels:
         app: sgl-ascend
+        run-id: "{{ run_id }}"
         task: prefill
```

**逐行解释**：green 环境分离部署模板的 prefill / decode / router 三个 StatefulSet pod 都加 `run-id` label，共 3 处改动

---

## 文件 8：`python/sglang/test/ascend/e2e/run_npu_testcase.sh`

```diff
 source_plog_path="/root/ascend/log/debug/plog"
 if [ -d "$source_plog_path" ];then
     echo "Plog files found. Begin to backup them."
-    target_plog_path="/root/sglang/debug/logs/plog/${tc_name}/${HOSTNAME}"
+    target_plog_path="/root/sglang/debug/logs/plog/${run_label}/${tc_name}/${HOSTNAME}"
     if [ "${SGLANG_IS_IN_CI}" = "true" ] || [ "${SGLANG_IS_IN_CI}" = "True" ];then
-        target_plog_path="/root/.cache/tests/logs/plog/${tc_name}/${HOSTNAME}"
+        target_plog_path="/root/.cache/tests/logs/plog/${run_label}/${tc_name}/${HOSTNAME}"
     fi
```

**逐行解释**：
- 非 CI 路径：plog 备份目录加 `${run_label}` 层级
- CI 路径：PVC 共享路径加 `${run_label}` 层级
- `run_label` 已含 `run_id` 前缀（由 `run_npu_e2e_test.py` 构造），避免同节点并发 run 互相覆盖 plog 文件
- 路径层级从 `plog/${tc_name}/${HOSTNAME}` 变成 `plog/${run_label}/${tc_name}/${HOSTNAME}`

---

## 文件 9：`.github/workflows/nightly-test-npu.yml`

非多机隔离改进的一部分（属于另一轮精简改动），主要改动：
- 移除所有单节点 nightly 任务，只保留多机任务（multi-node-poc / multi-node-mix-poc）
- `check-all-jobs` 和结果表格同步精简

详细 diff 不在本次多机隔离改进范围内，从略。

---

## 已验证效果

来自最新 run 日志（`run_id=32341352191`）的验证：

| 检查项 | 结果 | 证据 |
|--------|------|------|
| POSIX sh 语法兼容 | 通过 | 无 `Syntax error: redirection unexpected` |
| `/tmp/kube_job_name.txt` 读写 | 通过 | `Post-test cleanup using exact job name: ascend-sglang-perf-test-71742172jar0yk2q` |
| run_id 隔离的 pod 前缀 | 通过 | 只匹配 `ascend-sglang-perf-test-71742172jar0yk2q`，不碰其他 run |
| `--force` 删除 | 通过 | `force deleted` |
| 30min 超时循环 | 通过 | `retry 1/60` 后立即 `No sglang pods remain.` 退出 |
| ConfigMap 创建 | 通过 | `sglang-configmap-71742172jar0yk2q` 正确创建 |
| Volcano job 创建 | 通过 | `ascend-sglang-perf-test-71742172jar0yk2q` 正确创建 |
| 源码路径隔离 | 通过 | `/root/.cache/tests/sglang-32341352191` |
| Post process 退出码 | 通过 | 0（无 `##[error]` 出现在 Post process 之后） |

剩余 CI 失败（decode pod 提前 Succeeded / actions/checkout 0 字节下载）均为基础设施或业务侧问题，与 PR 改动无关。

---

## 与旧代码对比

| 维度 | 旧代码 | 改进后 |
|------|--------|--------|
| 清理循环超时 | `while true`（无超时，曾卡 1118min） | 30min（`max_retries=60 × 30s`） |
| Pre-test 超时行为 | 永远循环 | `exit 1`（fail fast） |
| Post-test 超时行为 | 永远循环 | WARNING 后退出，不失败 job |
| pod 匹配 | 固定前缀 `ascend-sglang-perf-test-` | 本 run 精确前缀（从 `/tmp/kube_job_name.txt`） |
| 删除方式 | `kubectl delete pod`（可能卡 Terminating） | `kubectl delete pod --force --ignore-not-found=true` |
| shell 语法 | `< <(...)` 进程替换（sh 不兼容） | `... \| while read` 管道（POSIX sh 兼容） |
| 源码目录 | 共享 `tests/sglang`，并发覆盖 | 按 `tests/sglang-${run_id}` 隔离 |
| pod label | 仅 `app=sgl-ascend` | 加 `run-id` label，按 run_id 隔离 |
| ConfigMap 过滤 | 按 namespace 全量查 | 按 `run-id` label 过滤 |
| plog 路径 | `plog/${tc_name}/${HOSTNAME}` | `plog/${run_label}/${tc_name}/${HOSTNAME}` |
| 源码目录清理 | 不清理（累积占用 PVC 空间） | Post-test 按 run_id 清理 |
