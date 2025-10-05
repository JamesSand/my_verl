# Zhizhou: we can merge pissa same as merge lora
# Because the only difference between pissa and lora is the way they init
# TODO: Need to figure out what the lora alpha meaning in pissa. This is important because we need to understand how extractly it should be merged.

# Zhizhou: The following code is adapted from
# https://github.com/volcengine/verl/pull/1821#issuecomment-3150776590 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 根目录：输入（FSDP合并后）与输出（LoRA融合后）
SRC_ROOT = Path("checkpoints_fsdp_merged")   # 读取这里
DST_ROOT = Path("checkpoints_lora_merged")   # 保存到这里

def collect_model_dirs(src_root: Path):
    """
    收集所有满足：<something>/global_step_100 下存在 lora_adapter/ 的目录。
    返回这些 global_step_100 目录的列表。
    """
    candidates = []
    # 只匹配名为 global_step_100 的目录，速度更快也更安全
    for p in src_root.rglob("global_step_100"):
        if not p.is_dir():
            continue
        if (p / "lora_adapter").is_dir():
            candidates.append(p)
    return sorted(candidates)

def main():
    # 收集待处理目录
    to_process = collect_model_dirs(SRC_ROOT)
    total = len(to_process)
    if total == 0:
        print(f"没有在 {SRC_ROOT} 下找到包含 lora_adapter 的 global_step_100 目录，退出。")
        return

    merged_cnt = 0
    skipped_cnt = 0
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"共 {total} 个待处理目录。输出将写入：{DST_ROOT.resolve()}\n")

    for idx, model_dir in enumerate(tqdm(to_process, desc="Merging LoRA", unit="model")):
        lora_adapter_dir = model_dir / "lora_adapter"

        # 计算相对路径并映射到目标根目录（保持结构一致）
        rel_path = model_dir.relative_to(SRC_ROOT)  # e.g. grpo-gsm8k/qwen2.5-0.5b-.../global_step_100
        target_dir = DST_ROOT / rel_path            # 同一层级到 checkpoints_lora_merged 下

        # 目标已存在 -> 跳过
        if target_dir.exists():
            skipped_cnt += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(model_dir.as_posix())  
        pmodel = PeftModel.from_pretrained(model, lora_adapter_dir.as_posix())  
        pmodel = pmodel.merge_and_unload() # Merging LoRA weights for vLLM compatibility

        # 保存到目标目录（与源目录同名结构）
        pmodel.save_pretrained(target_dir.as_posix())
        merged_cnt += 1

    print("\n完成：")
    print(f"  总计    : {total}")
    print(f"  已合并  : {merged_cnt}")
    print(f"  已跳过  : {skipped_cnt}")
    print(f"输出根目录: {DST_ROOT.resolve()}")

if __name__ == "__main__":
    main()






















# from transformers import AutoModelForCausalLM
# from peft import PeftModel
# import os
# from tqdm import tqdm

# model_dir_list = [
#     "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-lora32-lr3e-5/global_step_100",
#     "/ssd2/zhizhou/workspace/verl/checkpoints_merged/grpo-gsm8k/qwen2.5-0.5b-pissa32-lr3e-5/global_step_100"
# ]

# for model_dir in model_dir_list:

#     lora_adapter_dir = os.path.join(model_dir, "lora_adapter")

#     # if model dir not exists, raise
#     if not os.path.exists(model_dir):
#         raise ValueError(f"Model dir {model_dir} does not exist")
#     if not os.path.exists(lora_adapter_dir):
#         raise ValueError(f"LoRA adapter dir {lora_adapter_dir} does not exist")

#     model = AutoModelForCausalLM.from_pretrained(model_dir)  
#     pmodel = PeftModel.from_pretrained(model, lora_adapter_dir)  
#     pmodel = pmodel.merge_and_unload() # Merging LoRA weights for vLLM compatibility

#     lora_merged_dir = model_dir + "-lora_merged"
#     os.makedirs(lora_merged_dir, exist_ok=True)
#     pmodel.save_pretrained(lora_merged_dir)
#     print(f"Merged model saved to {lora_merged_dir}")


