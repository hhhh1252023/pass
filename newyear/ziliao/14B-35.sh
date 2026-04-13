echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --trust-remote-code \
    --attention-backend ascend \
    --disable-radix-cache \
    --mem-fraction-static 0.8 \
    --tp-size 2 --dp-size 1 \
    --nnodes 1 --node-rank 0 \
    --host 127.0.0.1 \
    --port 7777 \
    --sampling-backend ascend \
    --max-running-requests 16 \
    --served-model-name Qwen3-14B \
    --chunked-prefill-size -1 \
    --cuda-graph-bs 16 \
    --quantization modelslim \
    --dtype bfloat16 \
    --speculative-draft-model-quantization unquant \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path xxx \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --schedule-conservativeness 0.01
