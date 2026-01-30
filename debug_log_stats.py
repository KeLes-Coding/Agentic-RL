import json
import os
import glob

log_dir = r"I:\Paper\260125\verl-agent\local_logger\20260130_171846"
files = glob.glob(os.path.join(log_dir, "worker_env_steps_*.jsonl"))

if not files:
    print("No worker files found!")
    exit()

fpath = files[0]
print(f"Reading {fpath}...")

max_step = 0
done_count = 0
total_lines = 0
step_counts = {}

with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        total_lines += 1
        try:
            data = json.loads(line)
            step = data.get('step_idx', -1)
            done = data.get('done', False)
            
            if step > max_step:
                max_step = step
            
            if done:
                done_count += 1
                
            step_counts[step] = step_counts.get(step, 0) + 1
        except:
            pass

print(f"Total Lines: {total_lines}")
print(f"Max Step Index: {max_step}")
print(f"Total Done=True Events: {done_count}")
print(f"Step Distribution (first 5 and last 5):")
sorted_steps = sorted(step_counts.keys())
for s in sorted_steps[:5]:
    print(f"  Step {s}: {step_counts[s]}")
print("  ...")
for s in sorted_steps[-5:]:
    print(f"  Step {s}: {step_counts[s]}")
