# CCAPO v4.0 Runnable Fix Checklist

This checklist tracks the concrete code/config changes required to run CCAPO as dual-stream optimization instead of "GRPO + modified reward".

## 1) Advantage Path

- [x] Switch training estimator to `algorithm.adv_estimator=ccapo`.
- [x] Fix `CCAPO` branch crash in trainer (`token_level_rewards` undefined fallback path).
  - File: `verl/trainer/ppo/ray_trainer.py`

## 2) Macro Reward (`R_tau`) Semantics

- [x] Use v4.0 style macro reward decomposition:
  - `R_tau = I(success) * r_terminal + n_steps * r_penalty + I(failure) * r_failure`
  - File: `agent_system/ccapo/manager.py`
- [x] Apply time penalty on both success/failure trajectories.
- [x] Set v4.0-aligned defaults:
  - `r_terminal=10.0`
  - `r_penalty=-0.05`
  - `r_failure=0.0`
  - File: `agent_system/ccapo/config.py`

## 3) Remove Reward Double Counting

- [x] Do not add env terminal reward and `R_tau` together.
  - Use `rewards[i] = episode_result["r_tau"]` on episode end.
  - File: `agent_system/environments/env_manager.py`

## 4) Micro Stream (`A_micro`) Stability

- [x] Harden conversion of `a_micro_raw` from `non_tensor_batch` object arrays.
  - File: `agent_system/ccapo/ccapo_advantage.py`
- [x] Keep step-aligned micro advantages as rollout payload.

## 5) STDB Scoring Alignment

- [x] Align `I(E)` and global baseline counters to v4.0 edge-level statistics:
  - `I(E) = N_succ(E) / N_succ_total`
  - `C(E) = P(S|E) / P(S_global)` (edge-level)
  - File: `agent_system/ccapo/stdb.py`
- [x] Add global counters:
  - `total_success_edges`
  - `total_edges`

## 6) Novelty Bonus Control

- [x] Make novelty bonus optional and disabled by default for v4.0 behavior.
  - `novelty_bonus_coef=0.0`
  - Files: `agent_system/ccapo/config.py`, `agent_system/ccapo/manager.py`, `agent_system/environments/env_manager.py`

## 7) Runnable Script (Mode 2)

- [x] Update script to truly run CCAPO and only pass supported STDB params.
  - File: `examples/ccapo_trainer/run_mode_2_reward.sh`
- [x] Added cold-start seed path:
  - `++algorithm.ccapo.stdb.seed_path="$(pwd)/stdb_cold_start_output/alfworld_cold_start.json"`

## 8) Quick Sanity Run

Use:

```bash
bash examples/ccapo_trainer/run_mode_2_reward.sh
```

Expected preconditions:

- 2 visible GPUs (script currently uses `CUDA_VISIBLE_DEVICES="1,2"`).
- ALFWorld dependencies installed.
- Model path exists:
  - `/home/zzh/Workspace/modelscope/models/Qwen/Qwen2.5-1.5B-Instruct`
