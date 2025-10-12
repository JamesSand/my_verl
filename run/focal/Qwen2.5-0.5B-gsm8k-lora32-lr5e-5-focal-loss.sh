# export RAY_DEBUG_POST_MORTEM=1

set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# this training setting is from 
# https://github.com/thelongestusernameofall/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz64_2-prompt512-resp1024-lorarank32-score0.543.log

# export CUDA_VISIBLE_DEVICES=0,1

nproc_per_gpu=32
nnodes=1
ngpu_per_node=2
total_procs=$(( nproc_per_gpu * nnodes * ngpu_per_node ))

# Zhizhou: I cannot set mini batch size to 64, that is too large. Need set it small.
# mini_batch_size=$(( total_procs ))

learning_rate=5e-5
kl_coef=0.0
lora_rank=32
lora_alpha=32

MODEL_PATH=/ssd2/zhizhou/workspace/verl/models/Qwen2.5-0.5B-Instruct
model_basename=$(basename $MODEL_PATH)

train_files="[data/weqweasdas_from_default_filtered_openr1/train_0_10000.parquet]"
val_files="[data/weqweasdas_olympiadbench/test.parquet,data/weqweasdas_math500/test.parquet,data/weqweasdas_aime_hmmt_brumo_cmimc_amc23/test.parquet]"


# focal loss settings
actor_loss_agg_mode=focal-loss
focal_gamma=2.0
focal_beta_pos=1.0
focal_beta_neg=1.0

# input output settings
max_prompt_length=$((1024 * 1))
max_response_length=$((2048))
max_model_length=$((max_prompt_length + max_response_length))

offload=False

WANDB_PROJECT=focal-${model_basename}
# WANDB_EXP=debug

# 公共前缀（不含 focal 专属参数）
base_exp="${model_basename}-xiongdataset-lora${lora_rank}-lr${learning_rate}-kl${kl_coef}-${actor_loss_agg_mode}"

# 只有 focal-loss 才追加 gamma/beta
if [[ "${actor_loss_agg_mode}" == "focal-loss" ]]; then
  WANDB_EXP="${base_exp}-gamma${focal_gamma}-beta_pos${focal_beta_pos}-beta_neg${focal_beta_neg}"
else
  WANDB_EXP="${base_exp}"
fi


# WANDB_EXP=${model_basename}-xiongdataset-lora${lora_rank}-lr${learning_rate}-kl${kl_coef}-${actor_loss_agg_mode}-gamma${focal_gamma}-beta_pos${focal_beta_pos}-beta_neg${focal_beta_neg}

BASE_DIR=/ssd2/zhizhou/workspace/verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=${total_procs} \
    data.val_batch_size=${total_procs} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.model.use_shm=True  \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=${actor_loss_agg_mode} \
    actor_rollout_ref.actor.focal_gamma=${focal_gamma} \
    actor_rollout_ref.actor.focal_beta_pos=${focal_beta_pos} \
    actor_rollout_ref.actor.focal_beta_neg=${focal_beta_neg} \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=${max_model_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.total_training_steps=120 \
    trainer.total_epochs=666 $@ 2>&1 | tee ${WANDB_EXP}.log


