
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:7239 \
    --decode  http://127.0.0.1:8239 \
    --host 127.0.0.1 \
    --port 11000 \
    --health-check-interval-secs 3600
