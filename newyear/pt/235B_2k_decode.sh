
# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export ASCEND_MF_STORE_URL="tcp://141.61.105.141:24667"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DP_ROUND_ROBIN=1
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/root/.cache/modelscope/hub/models/hot_map

MODEL_PATH=/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8
EXPERTS_PATH=/root/.cache/modelscope/hub/models/hot_map/235B_2k_decode.pt
EAGLE3_PATH=/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=144
export HCCL_BUFFSIZE=512
export HCCL_SOCKET_IFNAME=enp48s3u1u1
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export STREAMS_PER_DEVICE=32
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_PROFILING_BS=36
export FUSED_DEEP_MOE_MODE=2


python  -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
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
