export PYTHONPATH=/data/l30079981/run_sglang_debug/run_sglang/1224:$PYTHONPATH
# 后台执行所有 Ascend 测试，日志保存到 /data/ascend_all_logs
mkdir -p /data/ascend_all_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
nohup python run_suite.py \
  --suite all \
  --continue-on-error \
  --enable-retry \
  --timeout-per-file 3600 \
  --log-dir /data/ascend_all_logs \
 | split -b 10M -d -a 3 - /data/ascend_all_logs/console_${TIMESTAMP}_part_2>&1 &
 # > ${LOG_FILE} 2>&1 &
  
