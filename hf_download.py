
from huggingface_hub import snapshot_download

# model_name = "Qwen/Qwen3-8B-Base"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

local_dir = snapshot_download(
    repo_id=model_name,
    local_dir=f"/ssd2/zhizhou/workspace/verl/models/{model_name.split('/')[-1]}",
    local_dir_use_symlinks=False
)
print("model saved to:", local_dir)


