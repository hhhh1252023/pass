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

#!/bin/bash

# 第一步：等待10秒
echo "等待10秒后开始发送请求..."
sleep 10

# 定义测试用例：长度→精准字符文本（严格匹配[10,20,30,40,45,57]）
declare -A test_cases=(
    [10]="a b c d e f"                # 10字符
    [20]="a b c d e f g h i j k l"    # 20字符
    [30]="a b c d e f g h i j k l m n o p q r s t"  # 30字符
    [40]="a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d"  # 40字符
    [45]="a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i"  # 45字符
    [57]="a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g"  # 57字符
)

# 接口地址
API_URL="http://127.0.0.1:30000/generate"

# 第二步：批量发送请求
echo "开始发送请求..."
for len in "${!test_cases[@]}"; do
    text="${test_cases[$len]}"
    echo "发送字符长度 $len 的请求..."
    
    # 仅发送请求，无额外操作
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

echo "所有请求发送完毕！"
