# 单机混布
# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
export PYTHONPATH=${PWD}/python:$PYTHONPATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# pd传输, IP设置为p节点首节点
export ASCEND_MF_STORE_URL="tcp://172.22.3.71:24580"

# p节点IP
P_IP=('172.22.3.71')
# D节点IP D节点首节点IP
D_IP=('172.22.3.166' '172.22.3.181')

MODEL_PATH=/home/weights/Qwen/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"
# prefill
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        export HCCL_BUFFSIZE=1200
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2

        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo

        export SGLANG_NPU_FUSED_MOE_MODE=2
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=262144


        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \
        --port 8000 --disaggregation-bootstrap-port $((8995+$i)) --trust-remote-code --nnodes 1 --node-rank 0 \
        --tp-size 16 --mem-fraction-static 0.7 --attention-backend ascend --device npu --quantization modelslim \
        --disaggregation-transfer-backend ascend --max-running-requests 24 --disable-radix-cache \
        --chunked-prefill-size 16384 --max-prefill-tokens 32768 --moe-a2a-backend ascend_fuseep \
        --ep-dispatch-algorithm static --init-expert-location /home/weights/hot_map/480_2k_prefill.pt \
        --dp-size 2 --enable-dp-attention --dtype bfloat16 --disable-overlap-schedule
        NODE_RANK=$i
        break
    fi
done


# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
    echo "${D_IP[$i]}"
    export HCCL_BUFFSIZE=600
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536
    export HCCL_SOCKET_IFNAME=data0.3001
    export GLOO_SOCKET_IFNAME=data0.3001
     export SGLANG_NPU_PROFILING=0
     export SGLANG_NPU_PROFILING_BS=160
    export SGLANG_NPU_FUSED_MOE_MODE=2

	python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8001 --trust-remote-code --dist-init-addr ${D_IP[0]}:5000 --nnodes 2 --node-rank $i --tp-size 32 --dp-size 4 \
        --mem-fraction-static 0.75 --max-running-requests 640 --attention-backend ascend --device npu --quantization modelslim \
        --moe-a2a-backend ascend_fuseep --enable-dp-attention --enable-dp-lm-head \
        --ep-dispatch-algorithm static --init-expert-location /home/weights/hot_map/480_2k_decode.pt \
        --cuda-graph-bs 48 64 72 96 112 120 128 136 144 152 160 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --tokenizer-worker-num 4 --prefill-round-robin-balance --dtype bfloat16  --load-balance-method round_robin
        NODE_RANK=$i
        break
    fi
done

exit 1

