#!/bin/bash

# 定义测试用例：长度→精准字符文本（字符数严格匹配[5,10,15,20,24,26]）
declare -A test_cases=(
    [5]="a b c d"       # 5字符
    [10]="a b c d e f"  # 10字符
    [15]="a b c d e f g h i"  # 15字符
    [20]="a b c d e f g h i j k l"  # 20字符
    [24]="a b c d e f g h i j k l m n o"  # 24字符
    [26]="a b c d e f g h i j k l m n o p q"  # 26字符
)

# 接口地址
API_URL="http://127.0.0.1:30000/generate"

# 循环发送请求
for len in "${!test_cases[@]}"; do
    text="${test_cases[$len]}"
    echo "发送字符长度 $len 的请求..."
    
    # 仅发送请求，不保存响应/日志
    curl -s --location "$API_URL" \
    --header 'Content-Type: application/json' \
    --data "{
        \"text\": \"$text\",
        \"sampling_params\": {
            \"temperature\": 0,
            \"max_new_tokens\": 10
        }
    }"
    
    echo "长度 $len 请求发送完成！"
    echo "-------------------------"
done
