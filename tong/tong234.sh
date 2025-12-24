echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_goveror
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=500000000
export SGLANG_SET_CPU_AFFINITY=1
export PYTHONPATH=/home/syy/Sglang/0729/sglang_npu/python:$PYTHONPATH
export GLOO_SOCKET_IFNAME=enp23s0f3
export HCCL_SOCKET_IFNAME=enp23s0f3
export HCCL_BUFFSIZE=600
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export ENBALE_PROFILING=0
export ASCEND_MF_STORE_URL="tcp://192.168.0.124:24666"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=48


USE_VLLM_CUSTOM_ALLREDUCE=1 python -m sglang.launch_server --model-path /home/qdl/dsv --disaggregation-ib-device mlx5_0,mlx5_1 --disaggregation-mode prefill --host 192.168.0.234 --port 8000 --trust-remote-code --tp-size 16 --dp-size 2 --enable-dp-attention --mem-fraction-static 0.7 --attention-backend ascend --device npu --quantization w8a8_int8 --disaggregation-bootstrap-port 8090 --disaggregation-transfer-backend ascend
