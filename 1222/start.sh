#!/bin/bash
# 无自定义变量版：仅保留核心调用+保活，完全贴合你的原脚本
CHECK_INTERVAL=60  # 检查间隔（秒）

# 启动/重启监控（直接写死 server，需 node 就改这里）
restart_monitor() {
    pkill -f "bash ./monitor.sh server" > /dev/null 2>&1
    nohup bash ./monitor.sh server > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo "[$(date +%H:%M:%S)] 监控进程已启动/重启（PID：$MONITOR_PID）"
}

# 后台循环检查
check_monitor_loop() {
    while true; do
        if ! ps -p $MONITOR_PID > /dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] 监控进程已退出，重启中..."
            restart_monitor
        fi
        sleep $CHECK_INTERVAL
    done
}

# 核心流程
restart_monitor
check_monitor_loop &
nohup test1 &> log.log &
echo "[$(date +%H:%M:%S)] 测试用例已启动，监控保活已开启（间隔${CHECK_INTERVAL}秒）"
