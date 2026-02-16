#!/bin/bash
set -x

# ================= GPU 上锁配置 =================
export CUDA_VISIBLE_DEVICES="0,1,2,3"     # 你原来就�?0,1，这里保持一�?
GPUS_TO_LOCK=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ')

# 脚本退出时恢复 GPU 模式
function cleanup {
    echo "[cleanup] 脚本结束，恢�?GPU 为默认模�?.."
    for GPU_ID in $GPUS_TO_LOCK; do
        # 建议把持久化模式也关回去（可选）
        sudo nvidia-smi -i $GPU_ID -c DEFAULT || true
        # sudo nvidia-smi -i $GPU_ID -pm 0 || true
    done
    echo "[cleanup] 恢复完成�?
}

trap cleanup EXIT INT TERM

echo "[lock] �?GPU ${GPUS_TO_LOCK} 设置独占模式..."
for GPU_ID in $GPUS_TO_LOCK; do
    sudo nvidia-smi -i $GPU_ID -pm 1
    sudo nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS
done
echo "[lock] 设置完成�?
# =================================================

# --- 🔥 [Step 0] 自动清理 (防止僵尸进程) ---
pkill -f "verl.trainer.main_ppo" || true
pkill -f "ray::" || true
sleep 2
# ----------------------------------------

export VLLM_ATTENTION_BACKEND=XFORMERS
export SWANLAB_API_KEY="oB8w36PCJxKeqwif2ijWz"

# 模型路径 (请确保路径存�?
MODEL_PATH="/home/zzh/Workspace/modelscope/models/Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_PATH="/home/zzh/Workspace/modelscope/models/Qwen/Qwen2.5-VL-3B-Instruct"

# 确保 PYTHONPATH 包含当前目录�?ALFWorld 环境�?
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/agent_system/environments/env_package/alfworld

# ================= 配置区域 =================
# SAMPLE_SIZE=40  <-- 不再需要总数，直接指定分集大�?
DATA_SEED=42
# TRAIN_RATIO=0.8 <-- 不再需�?

TRAIN_BATCH_SIZE=16
VAL_BATCH_SIZE=128
GROUP_SIZE=8
EXPERIMENT_NAME="ccapo_alfworld_real_run1_48x4"
MAX_STEPS=50

# 新增：显式控制数据集大小
TRAIN_SET_SIZE=128            # 例如�?�?batch
VAL_SET_SIZE=$VAL_BATCH_SIZE  # 强制让验证集大小等于验证 Batch Size
# ===========================================

echo ">>> [1/2] Generating/Updating Real ALFWorld Data..."
python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$(pwd)/data/verl-agent" \
    --train_data_size $TRAIN_SET_SIZE \
    --val_data_size $VAL_SET_SIZE \
    --data_source alfworld \
    --ability alfworld

DATA_DIR="$(pwd)/data/verl-agent/text"

echo ">>> [2/2] Starting CCAPO Training with LoRA (4 GPUs)..."

# 确保日志目录存在
mkdir -p logger

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    actor_rollout_ref.rollout.load_format=safetensors \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    ++actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    ++algorithm.ccapo.enable_ccapo=true \
    ++algorithm.ccapo.r_loop_penalty=-0.5 \
    ++algorithm.ccapo.enable_update_then_evaluate=true \
    ++algorithm.ccapo.stdb_save_path="stdb/alfworld_stdb.json" \
    ++algorithm.ccapo.invalid_action_penalty.enable=true \
    ++algorithm.ccapo.invalid_action_penalty.penalty_value=-0.5 \
    ++algorithm.ccapo.stdb.c_explore=2.0 \
    ++algorithm.ccapo.stdb.alpha_prior=1.0 \
    ++algorithm.ccapo.stdb.beta_prior=1.0 \
    ++algorithm.ccapo.stdb.enable_tanh_gating=true \
    ++algorithm.ccapo.stdb.reward_scale=1.0 \
    ++algorithm.ccapo.stdb.reward_temp=1.0 \
    ++algorithm.ccapo.beta_micro=0.5 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=42 \
    env.max_steps=$MAX_STEPS \
    env.rollout.n=$GROUP_SIZE \
    trainer.logger='[console,swanlab]' \
    trainer.project_name='verl_ccapo_debug' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=6 \
    trainer.val_before_train=True \
    2>&1 | tee logger/ccapo_run.log
