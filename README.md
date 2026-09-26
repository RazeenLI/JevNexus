# DeMa — Decision-based Schema Matching

这是一个零样本 schema matching 实验库。代码、数据、模型、运行状态、日志和最终指标已经按用途分开；仓库不再使用含义模糊的 `outputs/`。

## 目录规则

```text
configs/                 实验、模型、路径和扩展性配置
data/
  raw/                   下载/解压后的原始 benchmark
  processed/             统一格式后的实验输入
  manifests/             case 清单与文件校验信息
models/checkpoints/      本地模型权重（不提交 Git）
src/dema/
  data/                  下载、整理、加载数据
  baselines/             COMA、COMA++、Magneto、ISResMat、Unicorn 等
  model/                 DeMa 模型、检索、排序与列表示
  metrics/               指标定义、聚合和 evaluator
  experiments/           runner、preflight、smoke、scalability、precompute
  serving/               Qwen 的 Transformers fallback server
  utils/                 配置、I/O、日志等通用代码
vendor/magneto/          固定版本的 Magneto 上游源码及最小本地补丁
cache/candidates/        可删除、可重建的 DeMa 原始候选检索缓存
cache/candidates_magneto/ DeMa 对照变体共用的 Magneto 候选缓存
saves/                   可恢复实验所需的结构化状态和预测
logs/                    case、lane、server 的人类可读日志
metrics/                 最终 CSV 指标表
scripts/                 统一的运行入口
tests/                   无 GPU 的快速测试和可选慢测试
```

各运行目录的生命周期不同：

- `cache/` 可以随时删除并重算。
- `logs/` 只用于排错，不决定一个 case 是否完成。
- `saves/` 包含恢复实验所需的 prediction、runtime、status 和 run manifest。
- `metrics/` 由 evaluator 从 `saves/` 重建。

## 快速开始

仓库实验代码默认使用 `~/miniconda3/envs/airdb/bin/python`；vLLM 单独安装在
`~/.venvs/vllm`，避免它固定的 Torch 版本污染实验环境。可用下面的脚本在新机器上创建：

```bash
bash scripts/setup_vllm.sh
```

正常实验只需要一个 Python 命令。数据不存在时会自动整理；`magneto_qwen` 和
DeMa 变体所需的服务会按需启动、通过真实请求后再运行，并在该方法结束后自动关闭：

```bash
source scripts/_env.sh
$PYTHON -m dema run --config configs/experiment_dev.yaml
$PYTHON -m dema run --config configs/experiment.yaml
```

旧命令仍是兼容入口，行为与上面的 Python 命令相同：

```bash
EXPERIMENT_CONFIG=configs/experiment_dev.yaml bash scripts/run_all.sh
bash scripts/run_all.sh
```

单独跑一个模型或一个小样本也使用同一入口：

```bash
source scripts/_env.sh
$PYTHON -m dema run --methods dema_fusion --datasets GDC --limit 1 --overwrite
$PYTHON -m dema run --methods coma --datasets OpenData
```

默认使用 GPU 0，服务首次启动可通过 `--server-timeout 3600` 留出模型下载时间。
`--gpus 1` 会让模型服务和实验 worker 共用 GPU 1；`--gpus 1 0` 会把模型服务放在
GPU 1、将 Magneto MPNet/DeMa 检索模型等进程内模型放在 GPU 0。旧参数
`--qwen-gpu`、`--decision-gpu` 仍然兼容，且会让对应方法的 worker 使用同一张卡。
`--verify-data` 才会重新执行较慢的全量原始数据完整性检查。高级调试入口仍保留在
`dema.experiments.*`。

双 GPU lane：

```bash
nohup bash scripts/run_lane.sh qwen 0 >/dev/null 2>&1 &
nohup bash scripts/run_lane.sh decision 1 >/dev/null 2>&1 &
tail -f logs/lanes/*/lane.log
```

## 方法边界

`magneto_qwen` 使用 `vendor/magneto` 中固定提交的上游 Magneto 全流程，包括其列编码、MPNet 候选检索和 LLM reranker。唯一功能补丁是把 `litellm` 调用替换成本地 OpenAI-compatible Qwen endpoint。它不与 DeMa 共享候选缓存。

主实验与消融使用以下固定矩阵：

| 方法 | Candidate retriever | Scoring / reranking | 用途 |
|---|---|---|---|
| `magneto_qwen` | 原版 Magneto | Qwen listwise Top-20 | baseline |
| `dema_fusion` | Magneto 候选 | `0.4 × Jev-Single + 0.6 × COMA++` | **DeMa 基础版本** |
| `dema_single` | Magneto 候选 | Jev 只查看当前候选 | 去除 COMA++ 的消融 |
| `dema_shared` | Magneto 候选 | Jev 查看全部 Top-20 | context 消融 |
| `dema_own_retrieval` | DeMa 原始检索 | Jev 查看全部 Top-20 | retrieval 消融 |

`dema_fusion`、`dema_shared` 和 `dema_single` 复用 Magneto 的数据清洗、MPNet 序列化、mixed
sampling、exact-name match、阈值、Top-20、候选顺序及 dataset-specific 编码；Jev
看到的列名和 sampled values 也与 Magneto Qwen prompt 一致且不额外加入 dtype。
`dema_fusion` 对同一输入另行计算 COMA++ 的 schema/instance 分数，只在 Magneto
Top-20 内用固定原始分数公式重排；COMA++ 双向筛选未选中的 pair 记为 0。该方法不含
训练参数或额外 LLM ranker。`dema_own_retrieval` 保留 DeMa 原始 profile（类型和频率
采样）用于隔离 retrieval 的影响。旧方法名 `dema` 仅为已有 single + own-retrieval
结果保留，不再列入默认实验。

## 当前基础结果

下面是 `experiment_dev.yaml` 当前选取的 29 个有效 case 的 case-macro 结果。
`dema_fusion` 由已经保存的 `dema_single` 与 COMA++ 候选级分数离线复算确认；正式运行
使用完全相同的固定公式。它不是逐 case 选择较优方法的 oracle。

| 方法 | MRR | Recall@GT | Hits@1 | Hits@5 | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | MAP | 平均时间/case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Magneto-Qwen | 0.9501 | 0.8259 | 0.9386 | 0.9658 | 0.9440 | 0.9594 | 0.9683 | 0.9500 | 0.9431 | 133.72s |
| **DeMa-Fusion** | **0.9343** | **0.8050** | **0.9097** | **0.9726** | **0.9583** | **0.9643** | **0.9683** | **0.9401** | **0.9287** | 约 17.1s¹ |
| COMA++ | 0.9162 | 0.7382 | 0.9078 | 0.9271 | 0.8938 | 0.8948 | 0.9000 | 0.9089 | 0.9037 | 0.50s |
| DeMa-Single | 0.8845 | 0.6778 | 0.8301 | 0.9580 | 0.9409 | 0.9643 | 0.9683 | 0.9023 | 0.8781 | 16.59s |

¹ 尚未对正式 `dema_fusion` runner 单独计时；17.1s 是 DeMa-Single 与 COMA++
现有平均时间顺序相加的估计值。

各数据集的 MRR：

| 方法 | ChEMBL | GDC | Magellan | OpenData | TPC-DI | WikiData |
|---|---:|---:|---:|---:|---:|---:|
| Magneto-Qwen | 1.0000 | 0.7768 | 1.0000 | 0.9887 | 0.9933 | 0.9396 |
| **DeMa-Fusion** | **0.8604** | **0.8133** | **1.0000** | **0.9887** | **1.0000** | **0.9458** |
| COMA++ | 0.9758 | 0.5972 | 1.0000 | 0.9840 | 1.0000 | 0.9458 |
| DeMa-Single | 0.7894 | 0.7578 | 0.9217 | 0.9737 | 0.9933 | 0.8677 |

后续改进分为两条独立路线：基于置信度的动态融合权重（重点解决 ChEMBL），以及只对
低置信度 Top-3/Top-5 候选调用的小型 LLM ranker。

详细实现审计见：

- [Baseline audit](docs/BASELINE_AUDIT.md)
- [DeMa model and performance audit](docs/MODEL_AUDIT.md)

## 保存格式

```text
saves/predictions/<method>/<dataset>/<case_id>.json
saves/runtime/<method>/<dataset>/<case_id>.json
saves/status/<method>/<dataset>/<case_id>.json
saves/debug/<run_id>/<method>/<dataset>/<case_id>.jsonl
saves/manifests/<run_id>.json
saves/scalability/{raw,predictions}/...
saves/scalability/summary.csv

logs/cases/<method>/<dataset>/<case_id>.log
logs/lanes/<lane>/...

metrics/per_case.csv
metrics/per_dataset.csv
metrics/overall.csv
metrics/completeness.csv
```

一个 case 只有同时满足以下条件才会被 `--resume` 跳过：status 是 success、prediction hash 一致、runtime 存在，并且每个 source column 都有完整且无重复的 target ranking。失败或写到一半的 case 会重跑。

## 配置

- `configs/paths.yaml`：所有根目录和原始数据地址。
- `configs/models.yaml`：列表示、检索、模型服务和 baseline 参数。
- `configs/experiment.yaml`：正式 561-case 实验。
- `configs/experiment_dev.yaml`：每个数据集最多 5 个 case，用于迭代。
- `configs/scalability.yaml`：目标 schema 宽度实验，当前为每数据集 2 个 case、3 次重复。

不要直接用正式配置验证代码改动。先用 `experiment_dev.yaml` 或 `--limit 1`；完整实验会产生 13,522 个以上的 source-column 推理单元。

## 指标

所有方法统一由 `dema.metrics` 评估，模型代码不接触 ground truth。主要指标包括 MRR、Recall@GT、Hits@1/5/10、Recall@1/5/10/20、NDCG@5/10 和 MAP。`per_dataset.csv` 对 case 做宏平均，`overall.csv` 同时报告 case-macro 与 dataset-macro。

## 测试

```bash
source scripts/_env.sh
$PYTHON -m pytest
DEMA_SLOW_TESTS=1 $PYTHON -m pytest   # 会加载/下载 ISResMat 和 Unicorn 模型
```
