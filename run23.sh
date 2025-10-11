
# set -ex

export CUDA_VISIBLE_DEVICES=2,3

bash run/focal/Qwen2.5-0.5B-math-lora32-lr5e-6.sh
bash run/focal/Qwen2.5-0.5B-math-lora32-lr1e-6.sh

