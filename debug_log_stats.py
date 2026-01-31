import json
import os
import glob

log_dir = r"I:\Paper\260125\verl-agent\local_logger\20260130_171846"
files = glob.glob(os.path.join(log_dir, "worker_env_steps_*.jsonl"))

if not files:
    print("No worker files found!")
    exit()

# Sort by modification time
files.sort(key=os.path.getmtime, reverse=True)
fpath = files[0]
print(f"Reading newest file: {fpath}")

max_step = 0
done_count = 0
won_count = 0
total_lines = 0
step_counts = {}
rewards = []

with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        total_lines += 1
        try:
            data = json.loads(line)
            step = data.get('step_idx', -1)
            done = data.get('done', False)
            
            if step > max_step:
                max_step = step
            
            reward_ccapo = data.get('reward_ccapo', 0.0)
            won = data.get('won', False)
            
            if done:
                done_count += 1
            if won:
                won_count += 1
                
            rewards.append(reward_ccapo)
            step_counts[step] = step_counts.get(step, 0) + 1
        except:
            pass

avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
min_reward = min(rewards) if rewards else 0.0
max_reward = max(rewards) if rewards else 0.0

print(f"Total Lines: {total_lines}")
print(f"Max Step: {max_step}")
print(f"Done Events: {done_count}")
print(f"Won Events: {won_count}")
print(f"Reward CCAPO: Min={min_reward:.4f}, Max={max_reward:.4f}, Avg={avg_reward:.4f}")
print("Done.")
