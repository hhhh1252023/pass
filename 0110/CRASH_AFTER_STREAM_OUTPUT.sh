export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1
export ENABLE_ASCEND_MOE_NZ=1
export SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS=1

export PYTHONPATH=/home/q30061833/code/sglang/python:$PYTHONPATH

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --base-gpu-id 2 \
   --tp-size 2 \
   --model-path /home/weights/Qwen3-30B-A3B/Qwen3-30B-A3B \
   --port 8001 \
   --mem-fraction-static 0.8 \
   --dtype bfloat16 \
   --crash-dump-folder /home/q30061833/run_file/crash_dump_folder \
