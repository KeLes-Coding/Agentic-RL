# STDB 冷启动工具

使用外部LLM（如DeepSeek）生成ALFWORLD成功轨迹，用于STDB的冷启动。

## 功能特点

- ✅ **OpenAI SDK兼容** - 支持DeepSeek、OpenAI及其他兼容API
- ✅ **并发生成** - 使用ThreadPoolExecutor实现高效并发
- ✅ **与训练数据一致** - 支持从 `make_real_alfworld_data.py` 生成的parquet读取任务
- ✅ **完整轨迹记录** - 保存动作、观测、思考过程等详情
- ✅ **STDB格式导出** - 兼容现有 `stdb.seed_from_json()` 和 `stdb.merge_from_json()`

## 快速开始

### 1. 设置API Key

```bash
# DeepSeek
export DEEPSEEK_API_KEY="your-api-key"

# 或 OpenAI
export OPENAI_API_KEY="your-api-key"
```

### 2. 安装依赖

```bash
pip install openai tqdm pyyaml pandas pyarrow
```

### 3. 生成训练数据（如果没有）

```bash
# 与run_mode_0_grpo.sh一致的数据生成方式
python3 make_real_alfworld_data.py \
    --train_size 200 \
    --val_size 8 \
    --seed 42 \
    --output_dir data/verl-agent/text
```

### 4. 运行冷启动生成

```bash
# 推荐：从现有parquet读取（与训练数据一致）
python -m tools.stdb_cold_start.main \
    --data-source parquet \
    --parquet-path data/verl-agent/text/train.parquet \
    --model deepseek-chat \
    --workers 4

# 或：自动生成数据后读取
python -m tools.stdb_cold_start.main \
    --data-source generate \
    --samples-per-type 20

# 或：直接扫描ALFWORLD目录
python -m tools.stdb_cold_start.main \
    --data-source scan \
    --samples-per-type 20

# 快速测试
python -m tools.stdb_cold_start.main --test
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | - | YAML配置文件路径 |
| `--api-base` | https://api.deepseek.com/v1 | API Base URL |
| `--model` | deepseek-chat | 模型名称 |
| `--api-key` | 环境变量 | API Key |
| `--data-source` | parquet | 数据来源：parquet/generate/scan |
| `--parquet-path` | data/verl-agent/text/train.parquet | parquet文件路径 |
| `--samples-per-type` | 20 | 每种任务类型采样数量 |
| `--workers` | 4 | 并发worker数量 |
| `--max-steps` | 50 | 每个episode最大步数 |
| `--output-dir` | stdb_cold_start_output | 输出目录 |
| `--stdb-output` | stdb/alfworld_cold_start.json | STDB文件输出路径 |
| `--test` | - | 测试模式（快速运行） |

## 数据源模式

### parquet模式（推荐）

从 `make_real_alfworld_data.py` 生成的parquet文件读取任务，确保与训练数据完全一致：

```bash
python -m tools.stdb_cold_start.main \
    --data-source parquet \
    --parquet-path data/verl-agent/text/train.parquet
```

### generate模式

自动调用 `make_real_alfworld_data.py` 生成数据后读取：

```bash
python -m tools.stdb_cold_start.main --data-source generate
```

### scan模式

直接扫描ALFWORLD数据目录：

```bash
python -m tools.stdb_cold_start.main --data-source scan
```

## 输出文件

```
stdb/
└── alfworld_cold_start.json     # STDB seed文件（用于冷启动）

stdb_cold_start_output/
├── trajectories_full.json       # 完整轨迹详情
├── generation_summary.json      # 生成统计摘要
└── generation_*.log             # 运行日志
```

## 与STDB集成

### 方式1：作为seed文件

在training配置中指定：
```bash
ccapo.stdb.seed_path=stdb/alfworld_cold_start.json
```

### 方式2：叠加合并（推荐）

```python
from agent_system.ccapo.stdb import STDB
from agent_system.ccapo.config import STDBConfig

stdb = STDB(STDBConfig())
stdb.load("stdb/alfworld_stdb.json")  # 加载已有数据
stdb.merge_from_json("stdb/alfworld_cold_start.json")  # 叠加冷启动数据
stdb.save("stdb/alfworld_stdb.json")  # 保存合并后的STDB
```

## 配置文件示例

参考 `config_template.yaml`：

```yaml
api_base: "https://api.deepseek.com/v1"
model_name: "deepseek-chat"
data_source_mode: "parquet"
parquet_path: "data/verl-agent/text/train.parquet"
samples_per_type: 20
max_concurrent_workers: 4
```

## ALFWORLD任务类型

| 任务类型 | 说明 |
|----------|------|
| pick_and_place_simple | 拾取物品放置 |
| pick_clean_then_place_in_recep | 清洗后放置 |
| pick_heat_then_place_in_recep | 加热后放置 |
| pick_cool_then_place_in_recep | 冷却后放置 |
| look_at_obj_in_light | 在灯下查看物品 |
| pick_two_obj_and_place | 拾取两个物品放置 |

## 注意事项

1. **API限流**：根据API限制调整 `--workers` 数量
2. **ALFWorld环境**：确保ALFWorld已正确安装和配置
3. **成功率**：某些困难任务类型可能成功率较低，可增加 `--max-retries`
4. **数据一致性**：推荐使用 `--data-source parquet` 确保与训练数据一致
