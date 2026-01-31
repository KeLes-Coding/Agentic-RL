import os

fpath = r"I:\Paper\260125\verl-agent\logger\ccapo_run.log"
if not os.path.exists(fpath):
    print("File not found")
    exit()

with open(fpath, 'rb') as f:
    try:
        f.seek(-10000, os.SEEK_END)
    except:
        pass
    lines = f.readlines()
    for line in lines[-50:]:
        print(line.decode('utf-8', errors='ignore').strip())
