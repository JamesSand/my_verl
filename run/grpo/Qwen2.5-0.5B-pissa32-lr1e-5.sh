set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

export CUDA_VISIBLE_DEVICES=2,3

# pending for PID finish
WAIT_PID=1646118
tail --pid=$WAIT_PID -f /dev/null

nproc_per_gpu=32
nnodes=1
ngpu_per_node=2
total_procs=$(( nproc_per_gpu * nnodes * ngpu_per_node ))

# Zhizhou: I cannot set mini batch size to 64, that is too large. Need set it small.
# mini_batch_size=$(( total_procs ))

LR=1e-5

WANDB_PROJECT=grpo-gsm8k
WANDB_EXP=qwen2.5-0.5b-pissa32-lr${LR}

MODEL_PATH=/ssd2/zhizhou/workspace/verl/models/Qwen2.5-0.5B-Instruct

BASE_DIR=/ssd2/zhizhou/workspace/verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$BASE_DIR/data/gsm8k/train.parquet \
    data.val_files=$BASE_DIR/data/gsm8k/test.parquet \
    data.train_batch_size=${total_procs} \
    data.val_batch_size=${total_procs} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.model.use_shm=True  \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_init=pissa_niter_4 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@ 2>&1 | tee ${WANDB_EXP}.log