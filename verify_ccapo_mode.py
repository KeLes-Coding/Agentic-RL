import argparse
import json
import os
import numpy as np
import glob

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def verify_mode(log_dir):
    print(f"Verifying CCAPO Mode in: {log_dir}")
    
    # 1. Analyze Rewards (env level)
    rewards_file = os.path.join(log_dir, "detailed_rewards.jsonl")
    rewards_data = load_jsonl(rewards_file)
    
    has_micro_rewards = False
    has_stdb_correction = False
    
    if len(rewards_data) > 0:
        # Check last 50 episodes
        recent = rewards_data[-50:]
        micro_sum = sum([sum([s.get('r_stdb', 0.0) for s in ep.get('steps', [])]) for ep in recent])
        if abs(micro_sum) > 1e-5:
            has_micro_rewards = True
    
    print(f"[{'X' if has_micro_rewards else ' '}] CCAPO Micro Rewards (beta > 0)")
    
    # 2. Analyze LASR (training metrics)
    metrics_file = os.path.join(log_dir, "training_metrics.jsonl")
    metrics_data = load_jsonl(metrics_file)
    
    has_lasr = False
    lasr_stats = "N/A"
    
    if len(metrics_data) > 0:
        recent = metrics_data[-10:] # Last 10 steps
        # Check for lasr/weight_mean
        lasr_values = [m.get("lasr/weight_mean") for m in recent if "lasr/weight_mean" in m]
        if len(lasr_values) > 0:
            has_lasr = True
            lasr_stats = f"Mean={np.mean(lasr_values):.4f}"
            
    print(f"[{'X' if has_lasr else ' '}] LASR Weighting ({lasr_stats})")
    
    # 3. Infer Mode
    print("\n>>> INFERRED MODE:")
    if not has_micro_rewards and not has_lasr:
        print("  --> Mode 0: Baseline GRPO (or STDB Collection with beta=0)")
    elif has_micro_rewards and not has_lasr:
        print("  --> Mode 2: CCAPO Reward Only")
    elif not has_micro_rewards and has_lasr:
        print("  --> Mode 3: LASR Only (Unusual)")
    elif has_micro_rewards and has_lasr:
        print("  --> Mode 4: Full CCAPO (Reward + LASR)")
        
    print("\nStats Summary:")
    print(f"  Episodes Analyzed: {len(rewards_data)}")
    print(f"  Training Steps: {len(metrics_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logger", help="Log directory")
    args = parser.parse_args()
    verify_mode(args.log_dir)
