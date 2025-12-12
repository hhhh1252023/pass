pkill -9 sglang
pkill -9 python

#python3 -m sglang.launch_server --model-path /home/rjw/dsv3/weights \
#       --tp-size 16 --dp-size 1 --dist-init-addr 141.61.29.203:2245 --nnodes 2 --node-rank 0 \
#       --trust-remote-code --attention-backend triton --device npu --host 127.0.0.1 --port 2345 \
#       --disable-radix-cache --disable-overlap-schedule \
#       --quantization w8a8_int8 --disable-cuda-graph

export cann_path=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/driver/bin/setenv.bash
source ${cann_path}/../set_env.sh
source ${cann_path}/../../nnal/atb/set_env.sh
source ${cann_path}/opp/vendors/customize/bin/set_env.bash
export ASCEND_HOME_PATH=${cann_path}


# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑核
export SGLANG_SET_CPU_AFFINITY=1

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32


export HCCL_SOCKET_IFNAME=enp48s3u1u1
export GLOO_SOCKET_IFNAME=enp48s3u1u1


# hccl
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export HCCL_BUFFSIZE=1600
#export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_ALGO="level0:NA;level1:ring"


#export SGLANG_USE_MLAPO=1
#export SGLANG_NPU_USE_MLAPO=1
#source /usr/local/Ascend/ascend-toolkit/set_env.sh
#source /usr/local/Ascend/bisheng/latest/bisheng_toolkit/set_env.sh

#export ASCEND_LAUNCH_BLOCKING=1
export PYTHONPATH=$PWD/python/:$PYTHONPATH

#export SGLANG_DEBUG_MEMORY_POOL=true
python3 -m sglang.launch_server --model-path /mnt/share/l00850654/weights/DeepSeek-V3.2-W8A8 \
        --tp-size 16 --dp-size 1 \
        --trust-remote-code --attention-backend ascend --device npu --host 127.0.0.1 --port 2345 \
        --quantization w8a8_int8 --mem-fraction-static 0.79 \
	--chunked-prefill-size 64000 --context-length 66000 --max-prefill-tokens 66000 --max-total-tokens 66000 \
	--disable-radix-cache --moe-a2a-backend deepep --deepep-mode auto 
#--cuda-graph-bs 1 2 4 8 16 32
#--moe-a2a-backend deepep --deepep-mode auto
##--disable-cuda-graph --disable-radix-cache \
