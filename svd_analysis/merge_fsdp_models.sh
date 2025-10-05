
# This script is for merge model from fsdp
# after using this, please use merge_lora to merge lora weights

# This script will automatically find all actor model with global step 100 under checkpints folder, merge them and save to checkpoints_fsdp_merged folder

set -e


#!/usr/bin/env bash
# 用 global_step_100/actor 做 merge，结果写到 checkpoints_fsdp_merged/ 下，
# 目录层级与 checkpoints/ 下保持一致（去掉末尾的 /actor）

set -u

TARGET_ROOT="checkpoints_fsdp_merged"
mkdir -p "$TARGET_ROOT"

# 收集所有待处理的 actor 目录
mapfile -t ACTOR_DIRS < <(find checkpoints -type d -path "*/global_step_100/actor" | sort)
TOTAL=${#ACTOR_DIRS[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "nothing to merge, no actor dirs found under checkpoints/"
  exit 0
fi

MERGED=0
SKIPPED=0

for ((i=0; i<TOTAL; i++)); do
  LOCAL_DIR="${ACTOR_DIRS[$i]}"

  # 相对路径：去掉前缀 checkpoints/ 和末尾的 /actor
  REL_PATH="${LOCAL_DIR#checkpoints/}"
  REL_PATH="${REL_PATH%/actor}"

  TARGET_DIR="${TARGET_ROOT}/${REL_PATH}"

  # 进度提示（从 1 开始计数）
  echo "[$((i+1))/$TOTAL] 处理：$LOCAL_DIR"

  # 目标已存在则跳过
  if [ -d "$TARGET_DIR" ]; then
    echo "  -> 目标已存在，跳过：$TARGET_DIR"
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  mkdir -p "$TARGET_DIR"

  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

  echo "  -> 已保存到：$TARGET_DIR"
  MERGED=$((MERGED+1))
  echo "-----------------------------------------"
done

echo "完成：共 ${TOTAL} 个；合并 ${MERGED} 个，跳过 ${SKIPPED} 个。"





# # 在当前目录下运行：包含 checkpoints/ 目录
# TARGET_ROOT="checkpoints_fsdp_merged"

# mkdir -p "$TARGET_ROOT"

# # 找到所有 global_step_100 的 actor 目录
# while IFS= read -r LOCAL_DIR; do
#   echo "Merging from: $LOCAL_DIR"

#   # 取出在 checkpoints/ 之后的相对路径，并去掉末尾的 /actor
#   REL_PATH="${LOCAL_DIR#checkpoints/}"
#   REL_PATH="${REL_PATH%/actor}"

#   TARGET_DIR="${TARGET_ROOT}/${REL_PATH}"

#   mkdir -p "$TARGET_DIR"

#   python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir "$LOCAL_DIR" \
#     --target_dir "$TARGET_DIR"

#   echo "Saved to: $TARGET_DIR"
#   echo "-----------------------------------------"
# done < <(find checkpoints -type d -path "*/global_step_100/actor" | sort)

