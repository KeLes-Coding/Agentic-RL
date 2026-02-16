# CCAPO Trainer (48GB x 4 GPUs)

This folder contains 4-GPU presets adapted from `examples/ccapo_trainer`.

## Included Scripts

- `run_mode_0_grpo.sh`
- `run_mode_1_stdb.sh`
- `run_mode_2_reward.sh`
- `run_mode_3_lasr.sh`
- `run_alfworld_ccapo.sh`

## Key Preset Changes

- `CUDA_VISIBLE_DEVICES="0,1,2,3"`
- `trainer.n_gpus_per_node=4`
- `TRAIN_BATCH_SIZE=16`
- `VAL_BATCH_SIZE=128`
- `GROUP_SIZE=8`
- `actor_rollout_ref.actor.ppo_mini_batch_size=64`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8`
- `ACTOR_LR=1e-6` (where present)
- `actor_rollout_ref.actor.kl_loss_coef=0.01`
- `trainer.total_epochs=6`
- `trainer.val_before_train=True`
- Data preparation now uses official entry:
  `python3 -m examples.data_preprocess.prepare --mode text`
- For ALFWorld logging compatibility, scripts pass:
  `--data_source alfworld --ability alfworld`

## Usage

Run any script directly, for example:

```bash
bash examples/ccapo_trainer_48x4/run_mode_0_grpo.sh
```

Prerequisite: install Hugging Face `datasets` (`pip install datasets`) for `examples.data_preprocess.prepare`.
