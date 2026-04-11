pkill -9 python | pkill -9 sglang
#pkill -9 sglang
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH

#cd /home/chenxu/ifmn_sglang
export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

MODEL_PATH=/mnt/share/weights/Qwen3-Coder-480B-A35B-Instruct-W8A8
# pd传输, IP设置为p节点首节点
export ASCEND_MF_STORE_URL="tcp://172.22.3.71:24667"
# p节点IP
P_IP=('172.22.3.71')
# D节点IP
D_IP=('172.22.3.161' '172.22.3.181')
#export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DP_ROUND_ROBIN=1
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/mnt/share/chenxu/hot_map

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"


for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
        export HCCL_BUFFSIZE=2500
        export TASK_QUEUE_ENABLE=2
        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        export STREAMS_PER_DEVICE=32
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
#        export ENABLE_PROFILING=1

        # P节点
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode prefill \
        --host ${P_IP[$i]} --port 8000 --disaggregation-bootstrap-port 8995 --trust-remote-code \
        --nnodes 1 --node-rank $i --tp-size 16 --dp-size 2 --mem-fraction-static 0.6 \
        --disable-radix-cache \
        --expert-distribution-recorder-buffer-size -1 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static --enable-expert-distribution-metrics \
        --attention-backend ascend --device npu --quantization modelslim --disaggregation-transfer-backend ascend \
        --max-running-requests 128 --chunked-prefill-size 8192 --max-prefill-tokens 262144 \
        --enable-dp-attention  \
        --moe-a2a-backend deepep --deepep-mode normal --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done


for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=132
        export HCCL_BUFFSIZE=600
        export HCCL_SOCKET_IFNAME=data0.3001
        export GLOO_SOCKET_IFNAME=data0.3001
        export STREAMS_PER_DEVICE=32
        #export ENABLE_PROFILING=1

        # D节点
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
        --host ${D_IP[$i]} --port 8001 --trust-remote-code \
        --load-balance-method decode_round_robin \
        --nnodes 2 --node-rank $i --tp-size 32 --dp-size 4 --mem-fraction-static 0.73 --max-running-requests 512 \
        --attention-backend ascend --device npu --quantization modelslim --enable-dp-attention \
        --moe-a2a-backend deepep --deepep-mode low_latency --cuda-graph-bs 48 64 72 88 96 104 112 120 128 \
        --dist-init-addr 172.27.1.141:5000 \
        --expert-distribution-recorder-buffer-size -1 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static --enable-expert-distribution-metrics \
        --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --prefill-round-robin-balance --enable-dp-lm-head --dtype bfloat16 --tokenizer-worker-num 4
        NODE_RANK=$i
        break
    fi
done


#--disable-cuda-graph --moe-a2a-backend deepep --deepep-mode low_latency

exit 1
        --load-balance-method decode_round_robin \
        --ep-dispatch-algorithm static --init-expert-location /mnt/share/chenxu/hot_map/expert_distribution_recorder_1767503054.4450388.pt \

export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://172.22.3.71:8000 8995 \
    --decode http://172.22.3.161:8001 \
    --host 172.22.3.71 \
    --port 6688 \
    --mini-lb
python -m sglang.bench_serving --dataset-name random --backend sglang --host 141.61.105.145 --port 6688 --max-concurrency 768 --random-input-len 3500 --random-output-len 1500 --num-prompts 3072 --random-range-ratio 1
