# push_base_plus_adapter_and_merged.py
import os

# export HF token
# put hf token into bashrc

from huggingface_hub import HfApi

# local_file_path = "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-lora32-lr3e-5/global_step_100"

# api = HfApi(token=os.getenv("HF_TOKEN"))
# api.upload_folder(
#     folder_path=local_file_path,
#     repo_id="JameSand/qwen2.5_0.5b_lora32_lr3e_5_step100_base_and_lora_adapter",
#     repo_type="model",
# )

# local_file_path = "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-pissa32-lr3e-5/global_step_100"
# repo_id = "JameSand/qwen2.5_0.5b_pissa32_lr3e_5_step100_base_and_lora_adapter"


local_file_path = "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-pissa32-lr3e-5/global_step_100-lora_merged"
repo_id = "JameSand/qwen2.5_0.5b_pissa32_lr3e_5_step100_merged"

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=local_file_path,
    repo_id=repo_id,
    repo_type="model",
)

exit(0)

HF_USER = "JamesSand"   # 改成你的 HF 用户名
PRIVATE = False

BASE_DIR   = "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-pissa32-lr3e-5/global_step_100"
MERGED_DIR = BASE_DIR + "-lora_merged"

REPO_BASE_PLUS_ADAPTER = f"{HF_USER}/qwen2.5-0.5b-pissa32-lr3e-5-step100-base-and-lora_adapter"  # base + lora_adapte
REPO_MERGED            = f"{HF_USER}/qwen2.5-0.5b-pissa32-lr3e-5-step100-merged"     # merged model

# replace all - to _
REPO_BASE_PLUS_ADAPTER = REPO_BASE_PLUS_ADAPTER.replace("-", "_")
REPO_MERGED = REPO_MERGED.replace("-", "_")

print(REPO_BASE_PLUS_ADAPTER)
print(REPO_MERGED)

# breakpoint()

# api = HfApi()

# # 1) Repo#1：上传“base + lora_adapter”（整个 BASE_DIR）
# # api.create_repo(repo_id=REPO_BASE_PLUS_ADAPTER, repo_type="model", exist_ok=True, private=PRIVATE)
# api.upload_folder(folder_path=BASE_DIR, repo_id=REPO_BASE_PLUS_ADAPTER, repo_type="model")
# print(f"[OK] uploaded base+adapter -> https://huggingface.co/{REPO_BASE_PLUS_ADAPTER}")

# # 2) Repo#2：上传“合并后的整模”（MERGED_DIR）
# # api.create_repo(repo_id=REPO_MERGED, repo_type="model", exist_ok=True, private=PRIVATE)
# api.upload_folder(folder_path=MERGED_DIR, repo_id=REPO_MERGED, repo_type="model")
# print(f"[OK] uploaded merged -> https://huggingface.co/{REPO_MERGED}")
