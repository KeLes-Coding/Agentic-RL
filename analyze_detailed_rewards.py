import argparse
import json
import os
import numpy as np
from collections import defaultdict
import glob

def load_data(log_file):
    data = []
    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return data
        
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def analyze_episode(ep, gamma=0.99):
    steps = ep.get("steps", [])
    outcome = ep.get("outcome", False)
    
    total_r_macro = 0.0
    total_r_micro = 0.0 # STDB
    total_r_loop = 0.0
    total_r_invalid = 0.0
    total_r_valid = 0.0
    total_r_gross = 0.0 # sum of all components
    
    invalid_count = 0
    valid_count = 0
    
    discounted_return = 0.0
    
    # Calculate returns (backward pass for correct discounting if needed, 
    # but here we just want sums for component analysis)
    
    # Forward pass for sums
    for step in steps:
        total_r_micro += step.get("r_stdb", 0.0)
        total_r_loop += step.get("r_loop", 0.0)
        total_r_invalid += step.get("r_invalid", 0.0)
        total_r_valid += step.get("r_valid", 0.0)
        total_r_gross += step.get("r_total", 0.0)
        
        if not step.get("valid", False):
            invalid_count += 1
        else:
            valid_count += 1
            
    # Macro reward is usually implicit in the last step's total or injected.
    # In our manager.py logic, r_total includes everything.
    # Macro check: if outcome is true, last step should have huge boost?
    # Or strict definition: Macro = Outcome Reward.
    # Let's derive Macro by subtracting known components from Total?
    # Or just look at the raw components if we logged them?
    # We logged: r_loop, r_invalid, r_valid, r_stdb, r_total.
    # r_total = loop + invalid + valid + stdb + (Macro/Core if injected)
    # So Macro = r_total - (loop + invalid + valid + stdb)
    
    step_calculated_sum = sum([
        s.get("r_loop", 0.0) + s.get("r_invalid", 0.0) + s.get("r_valid", 0.0) + s.get("r_stdb", 0.0)
        for s in steps
    ])
    
    total_r_macro = total_r_gross - step_calculated_sum
    
    return {
        "len": len(steps),
        "outcome": outcome,
        "r_gross": total_r_gross,
        "r_macro": total_r_macro,
        "r_micro": total_r_micro,
        "r_penalty": total_r_loop + total_r_invalid,
        "invalid_cnt": invalid_count,
        "valid_rate": valid_count / len(steps) if len(steps) > 0 else 0.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="local_logger", help="Path to local_logger directory")
    args = parser.parse_args()
    
    # Find latest directory if not specific
    target_dir = args.log_dir
    if not os.path.exists(os.path.join(target_dir, "detailed_rewards.jsonl")):
        # Try finding subdirs
        subdirs = sorted(glob.glob(os.path.join(args.log_dir, "*")), reverse=True)
        for d in subdirs:
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "detailed_rewards.jsonl")):
                target_dir = d
                break
    
    log_file = os.path.join(target_dir, "detailed_rewards.jsonl")
    print(f"Analyzing log: {log_file}")
    
    episodes = load_data(log_file)
    print(f"Found {len(episodes)} episodes.")
    
    if not episodes:
        return

    stats = []
    for ep in episodes:
        stats.append(analyze_episode(ep))
        
    # Aggregate
    outcomes = [s["outcome"] for s in stats]
    success_rate = np.mean(outcomes)
    
    avg_len = np.mean([s["len"] for s in stats])
    avg_reward = np.mean([s["r_gross"] for s in stats])
    avg_macro = np.mean([s["r_macro"] for s in stats])
    avg_micro = np.mean([s["r_micro"] for s in stats])
    avg_penalty = np.mean([s["r_penalty"] for s in stats])
    avg_invalid = np.mean([s["invalid_cnt"] for s in stats])
    
    print("\n" + "="*40)
    print("       CCAPO Reward Analysis        ")
    print("="*40)
    print(f"Episodes Analyzed: {len(stats)}")
    print(f"Success Rate     : {success_rate:.2%}")
    print(f"Avg Length       : {avg_len:.2f}")
    print("-" * 40)
    print(f"Avg Total Reward : {avg_reward:.4f}")
    print(f"  - Macro (Goal) : {avg_macro:.4f}  ({avg_macro/avg_reward*100 if avg_reward!=0 else 0:.1f}%)")
    print(f"  - Micro (STDB) : {avg_micro:.4f}  ({avg_micro/avg_reward*100 if avg_reward!=0 else 0:.1f}%)")
    print(f"  - Penalties    : {avg_penalty:.4f}")
    print("-" * 40)
    print(f"Invalid Actions/Ep: {avg_invalid:.2f}")
    print("-" * 40)
    
    # Sample Display
    print("\nSample Episodes (Last 3):")
    for i, ep in enumerate(episodes[-3:]):
        s = stats[-3 + i]
        print(f"Ep {i+1} | Outcome: {s['outcome']} | Len: {s['len']} | R_Total: {s['r_gross']:.2f}")
        print(f"    Breakdown: Macro={s['r_macro']:.2f}, Micro={s['r_micro']:.2f}, Pen={s['r_penalty']:.2f}")
        print(f"    Invalid Actions: {s['invalid_cnt']}")
        
        # Show step details for last one
        if i == 2:
            print("    Trace Preview:")
            for j, step in enumerate(ep['steps'][:5]): # First 5
                print(f"      Step {j+1}: {step.get('action')[:40]}... -> Valid={step.get('valid')} | R_STDB={step.get('r_stdb'):.3f}")

if __name__ == "__main__":
    main()
