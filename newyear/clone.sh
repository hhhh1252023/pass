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
