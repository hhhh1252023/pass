export PYTHONPATH=/home/syy/Sglang/0729/sglang-npu/python:$PYTHONPATH


python -m sglang_router.launch_router --pd-disaggregation --prefill http://192.168.0.124:8000 8090 --prefill http://192.168.0.152:8000 8090 --prefill http://192.168.0.138:8000 8090 --decode http://192.168.0.173:8001 --prefill-policy bucket --balance-rel-threshold 1.0001 --balance-abs-threshold 32 --bucket-adjust-interval 10
