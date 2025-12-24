# 生成4097 token的文本（"a "重复4096次 + "b"）
TEXT=$(printf "a %.0s" {1..4096}; echo "b")

# 发送curl请求
curl --location 'http://127.0.0.1:30000/generate' \
--header 'Content-Type: application/json' \
--data "{
    \"text\": \"$TEXT\",
    \"sampling_params\": {
        \"temperature\": 0,
        \"max_new_tokens\": 10
    }
}"
