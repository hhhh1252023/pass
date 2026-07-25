git lfs install
# 第一个LoRA
git clone https://ghproxy.com/https://huggingface.co/nissenj/Qwen3-4B-lora-v2
# 第二个网文LoRA
git clone https://ghproxy.com/https://huggingface.co/TanXS/Qwen3-4B-LoRA-ZH-WebNovelty-v0.0

huggingface-cli download nissenj/Qwen3-4B-lora-v2 --resume-download --local-dir-use-symlinks False --local-dir ./Qwen3-4B-lora-v2

huggingface-cli download TanXS/Qwen3-4B-LoRA-ZH-WebNovelty-v0.0 --resume-download --local-dir-use-symlinks False --local-dir ./Qwen3-4B-LoRA-ZH-WebNovelty-v0.0
mkdir -p ~/.cache/datasets
wget -O ~/.cache/datasets/hellaswag_val.jsonl \
  https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl

wget https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3
https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png

hf download Styxxxx/llama2_7b_lora-trivia_qa --local-dir /data/models/llama2_7b_lora-trivia_qa --local-dir-use-symlinks False

# 1. 克隆 sgl-kernel-npu 仓库
cd d:\testcase
git clone https://github.com/sgl-project/sgl-kernel-npu.git
cd sgl-kernel-npu

# 2. 启动 CANN 9.0.0 容器（以 a3 硬件为例，根据你的实际镜像名调整）
docker run --rm -it -v ${PWD}:/workspace -w /workspace `
  quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11 `
  bash

# 3. 进入容器后执行：
#    先安装依赖（如果仓库内有 npu_ci_install_dependency.sh）
bash scripts/npu_ci_install_dependency.sh --torch-version 2.10.0

# 4. 编译 torch-memory-saver
env -i PATH=${PATH} bash --login -c "
  export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH} &&
  ./build.sh -a memory-saver
"

# 5. 产物在 output/ 目录，退出容器后自动消失（因为用了 --rm）
