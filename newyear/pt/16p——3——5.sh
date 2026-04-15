pkill -9 python | pkill -9 sglang
#pkill -9 sglang
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH


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

#export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

MODEL_PATH=/mnt/share/weights/Qwen3-235B-A22B-W8A8
# pd传输, IP设置为p节点首节点
export ASCEND_MF_STORE_URL="tcp://141.61.105.141:24667"
# p节点IP
P_IP=('172.22.3.209')
# D节点IP
D_IP=('172.22.3.181')
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
MODEL_PATH/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8
EXPERTS_PATH=/root/.cache/modelscope/hub/models/hot_map/235B_3_5k_bs26_decode.pt
EAGLE3_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3

for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
#        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
#        export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
        export HCCL_BUFFSIZE=2000
        export TASK_QUEUE_ENABLE=2
        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        export STREAMS_PER_DEVICE=32
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
#        export ASCEND_LAUNCH_BLOCKING=1
#        export ENABLE_PROFILING=1
        export FUSED_DEEP_MOE_MODE=2

        # P节点
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode prefill \
        --host ${P_IP[$i]} --port 8000 --disaggregation-bootstrap-port 8995 --trust-remote-code \
        --nnodes 1 --node-rank $i --tp-size 16 --dp-size 16 --mem-fraction-static 0.7 \
        --disable-radix-cache \
        --attention-backend ascend --device npu --quantization modelslim --disaggregation-transfer-backend ascend \
        --max-running-requests 128 --chunked-prefill-size 32768 --max-prefill-tokens 262144 \
        --enable-dp-attention --enable-dp-lm-head \
        --expert-distribution-recorder-buffer-size -1 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static --enable-expert-distribution-metrics \
        --speculative-algorithm EAGLE3 --speculative-draft-model-path $EAGLE3_PATH \
        --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2  --speculative-draft-model-quantization unquant  \
        --moe-a2a-backend ascend_fuseep --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done


for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=144
        export HCCL_BUFFSIZE=512
        export HCCL_SOCKET_IFNAME=enp48s3u1u1
        export GLOO_SOCKET_IFNAME=enp48s3u1u1
        export STREAMS_PER_DEVICE=32
        export SGLANG_NPU_PROFILING=0
        export SGLANG_NPU_PROFILING_BS=36
        export FUSED_DEEP_MOE_MODE=2

        # D节点
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
        --host ${D_IP[$i]} --port 8001 --trust-remote-code \
        --load-balance-method round_robin \
        --nnodes 1 --node-rank $i --tp-size 16 --dp-size 16 --mem-fraction-static 0.7 --max-running-requests 416 \
        --attention-backend ascend --device npu --quantization modelslim --enable-dp-attention \
        --moe-a2a-backend ascend_fuseep --cuda-graph-bs 1 2 4 8 16 20 24 26 \
        --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --speculative-algorithm EAGLE3 --speculative-draft-model-path $EAGLE3_PATH \
        --expert-distribution-recorder-buffer-size -1 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static --enable-expert-distribution-metrics \
        --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4  --speculative-draft-model-quantization unquant  \
        --prefill-round-robin-balance --enable-dp-lm-head --dtype bfloat16 --tokenizer-worker-num 4
        NODE_RANK=$i
        break
    fi
done


#--disable-cuda-graph --moe-a2a-backend deepep --deepep-mode low_latency
#--speculative-algorithm EAGLE3 --speculative-draft-model-path /mnt/share/weights/Qwen3-235B-A22B-Eagle3 \
#--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant \

exit 1
        --load-balance-method decode_round_robin \
        --ep-dispatch-algorithm static --init-expert-location /mnt/share/chenxu/hot_map/expert_distribution_recorder_1767503054.4450388.pt \

export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://141.61.105.143:8000 8995 \
    --decode http://141.61.105.145:8001 \
    --host 141.61.105.145 \
    --port 6688 \
    --mini-lb
python -m sglang.bench_serving --dataset-name random --backend sglang --host 141.61.105.145 --port 6688 --max-concurrency 768 --random-input-len 3500 --random-output-len 1500 --num-prompts 3072 --random-range-ratio 1
#        --speculative-algorithm EAGLE3 --speculative-draft-model-path /mnt/share/weights/Qwen3-235B-A22B-Eagle3 \
#        --speculative-draft-model-quantization unquant \
#        --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
# /mnt/share/chenxu/hot_map/expert_distribution_recorder_1767503054.4450388.pt


# eplb
0、添加以下参数
--expert-distribution-recorder-buffer-size -1 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static --enable-expert-distribution-metrics \

1、正常压测
python -m sglang.bench_serving --dataset-name random --backend sglang --host 145.61.105.145 --port 6688 --max-concurrency 768 --random-input-len 3500 --random-output-len 1500 --num-prompts 3072 --random-range-ratio 1
2、curl热度表，会生成一个pt文件
curl --location 'http://172.22.3.209:8000/dump_expert_distribution_record'
3、重启服务
修改--expert-distribution-recorder-buffer-size -1 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static --enable-expert-distribution-metrics \ 为
--ep-dispatch-algorithm static --init-expert-location /mnt/share/chenxu/hot_map/expert_distribution_recorder_1760671625.6899254.pt \
4、正常压测
python -m sglang.bench_serving --dataset-name random --backend sglang --host 145.61.105.145 --port 6688 --max-concurrency 768 --random-input-len 3500 --random-output-len 1500 --num-prompts 3072 --random-range-ratio 1

