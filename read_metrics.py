import json
import os

files = [
    r'i:\Paper\260125\verl-agent\training_metrics\00_GRPO.jsonl',
    r'i:\Paper\260125\verl-agent\training_metrics\02_CCAPO_reward.jsonl'
]

for f_path in files:
    print(f'\n--- {os.path.basename(f_path)} ---')
    try:
        if not os.path.exists(f_path):
            print("File not found.")
            continue
            
        with open(f_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Get last 5 lines
            tail_lines = lines[-5:] if len(lines) >= 5 else lines
            
            for line in tail_lines:
                try:
                    data = json.loads(line)
                    # Extract key metrics
                    metrics = {
                        "step": data.get("step"),
                        "success_rate": data.get("val/success_rate"),
                        "reward_mean": data.get("train/reward/mean"),
                        "reward_min": data.get("train/reward/min"),
                        "reward_max": data.get("train/reward/max"),
                        "reward_std": data.get("train/reward/std")
                    }
                    print(metrics)
                except json.JSONDecodeError:
                    print("JSON Decode Error")
    except Exception as e:
        print(f"Error reading file: {e}")
