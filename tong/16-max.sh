curl --location 'http://127.0.0.1:30000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "text": "a b c d e f g h i j k l m n o",  # 严格16字符
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 10
    }
}'

MAX_TEXT=$(printf "a %.0s" {1..4096})
curl --location 'http://127.0.0.1:30000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "text": "'"$MAX_TEXT"'",  # 4096字符（SGLang默认上限）
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 10
    }
}'
