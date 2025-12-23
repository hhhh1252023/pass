pkill -9 sglang
pkill -9 python

export cann_path=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/driver/bin/setenv.bash
source ${cann_path}/../set_env.sh
source ${cann_path}/../../nnal/atb/set_env.sh
source ${cann_path}/opp/vendors/customize/bin/set_env.bash
export ASCEND_HOME_PATH=${cann_path}


echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

export PYTHORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export HCCL_SOCKET_TFNAME=enp2133
export GLOO_SOCKET_IFNAME=enp2133

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export HCCL_BUFFSIZE=1600
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_ALGO="level0:NA;levrl1:ring"
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export STREAMS_PER_DEVICE=32
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export TASK_QUEUE_ENABLE=2
export PYTHONPATH=$PWD/python/:$PYTHONPATH


# P节点
python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode prefill \
--host ${P_IP[$i]} --port 8000 --disaggregation-bootstrap-port 8995 --trust-remote-code \
--nnodes 2 --node-rank $i --tp-size 16 --dp-size 16  --mem-fraction-static 0.6 \
--disable-radix-cache \
--ep-dispatch-algorithm static \
--attention-backend ascend --device npu --quantization w8a8_int8 --disaggregation-transfer-backend ascend \
--max-running-requests 128 --chunked-prefill-size 114688 --max-prefill-tokens 458880 \
--disable-overlap-schedule  --enable-dp-attention --tokenizer-worker-num 4 \
--moe-a2a-backend deepep --deepep-mode normal --dtype bfloat16 \
--dist-init-addr ${P_IP[0]}:5000
