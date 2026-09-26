# DeMa model and performance audit

## 已确认的耗时原因

正式 manifest 有 561 个 case、13,522 个 source columns。旧配置在每个 source column 上向 Open-Jev 提交 20 个二分类问题，但 `candidate_context: shared` 会让每个独立问题都重复携带全部 20 个候选；服务端同时使用 `--batch-size 1 --no-prefix-cache`。已有 DeMa runtime 记录显示 13,522 次请求共约 35,180 秒，其中 reranking 约 34,748 秒，平均每次约 2.57 秒。再叠加 Magneto/Qwen、神经 baseline 和旧 scalability 的大量重复，整套实验超过 24 小时是配置造成的，不是数据读取问题。

已做的修正：

- decision context 改为 `single`，每个问题只包含自己的 candidate；
- Open-Jev 默认 batch size 改为 20，并开启 prefix cache；
- candidate cache 只服务真正使用 DeMa retriever 的方法；
- dev 配置限制为每数据集 5 个 case；scalability 限制为每数据集 2 个 case、3 次重复；
- logs、saves、metrics、cache 分开，resume 不再依赖日志。

在同一张 RTX PRO 6000、同一个 Open-Jev 进程、20 个合成候选上实测：旧
`shared` 请求为 35,990 input tokens / 5.098 秒，新 `single` 请求为 4,270
input tokens / 0.482 秒。这个单次对照约减少 88% 输入 token、获得 10.6x
wall-time 加速；真实数据的加速比会随列值长度变化。

端到端 sanity case `magellan_amazon_google_exp`（4×4 columns）也已通过：新
DeMa 的 reranking 为 1.194 秒（4 requests、3,684 input tokens、无重试），
完整 matcher 时间 1.245 秒。相同 case 的 upstream-Magneto+Qwen 为 12.931 秒。
但该 case 上 DeMa MRR=0.875、Magneto MRR=1.0，再次说明性能问题已明显改善，
效果问题仍需通过训练或预先定义的 validation 策略解决。

服务端还报告 `causal_conv1d` 和 `flash-linear-attention` 未安装，因此当前
分别回退到 reference PyTorch kernel。结果是正确的，但速度会继续低于安装
兼容优化 kernel 的环境；安装前应先核对当前 CUDA、PyTorch 与模型版本。

Qwen server 若没有 vLLM 会退回 Transformers server。该 fallback 为保证可运行而设计，生成使用全局锁，吞吐量明显低于 vLLM；正式全量实验应优先安装并使用兼容版本的 vLLM。

## 当前主方法与消融

当前主方法 `dema` 保持 Magneto Top-20 候选，使用 Jev-Single 和 COMA++ 并行打分，
先按下面的固定原始分数公式融合，再由 `jinaai/jina-reranker-v3.5` 对 Top-3 做
listwise 重排：

```text
final_score = 0.4 × jev_score + 0.6 × coma_plus_score
```

COMA++ 经过双向筛选后未输出的 pair 记为 0，Top-20 之外的 retrieval tail 不变。公式
不按 dataset 分支；Jina revision 固定，且不在本 benchmark 上训练。

正式方法名如下。R 表示 Jina reranker，S 表示 COMA++ structured matcher：

| code | paper name | pipeline |
|---|---|---|
| `dema` | DeMa | Magneto candidates → Jev + COMA++ → Jina |
| `dema_no_rerank` | DeMa−R | Magneto candidates → Jev + COMA++ |
| `dema_no_struct` | DeMa−S | Magneto candidates → Jev → Jina |
| `dema_decision` | DeMa−R−S | Magneto candidates → Jev |
| `dema_shared` | DeMa-Shared | Magneto candidates → Jev shared-context |

当前 `experiment_dev.yaml` 29 个有效 case 的 case-macro 结果：

| method | MRR | Recall@GT | Recall@20 | NDCG@10 | MAP |
|---|---:|---:|---:|---:|---:|
| Magneto-Qwen | 0.9501 | 0.8259 | 0.9683 | 0.9500 | 0.9431 |
| **DeMa** | **0.9611** | **0.8020** | **0.9683** | **0.9593** | **0.9540** |
| DeMa−R | 0.9343 | 0.8050 | 0.9683 | 0.9401 | 0.9287 |
| DeMa−S | 0.9375 | 0.6990 | 0.9683 | 0.9409 | 0.9298 |
| COMA++ | 0.9162 | 0.7382 | 0.9000 | 0.9089 | 0.9037 |
| DeMa−R−S | 0.8845 | 0.6778 | 0.9683 | 0.9023 | 0.8781 |

所有四种方法都已通过统一 runner 完整运行。实测平均时间分别为 DeMa 17.62 秒/case、
DeMa−R 16.89 秒、DeMa−S 16.85 秒、DeMa−R−S 16.59 秒；Magneto-Qwen 为
133.72 秒/case。

各数据集 MRR 显示剩余问题主要集中在 ChEMBL：

| method | ChEMBL | GDC | Magellan | OpenData | TPC-DI | WikiData |
|---|---:|---:|---:|---:|---:|---:|
| Magneto-Qwen | 1.0000 | 0.7768 | 1.0000 | 0.9887 | 0.9933 | 0.9396 |
| DeMa | 1.0000 | 0.8477 | 1.0000 | 0.9944 | 0.9933 | 0.9240 |
| DeMa−R | 0.8604 | 0.8133 | 1.0000 | 0.9887 | 1.0000 | 0.9458 |
| DeMa−S | 0.9373 | 0.8230 | 0.9857 | 0.9804 | 1.0000 | 0.8885 |
| COMA++ | 0.9758 | 0.5972 | 1.0000 | 0.9840 | 1.0000 | 0.9458 |
| DeMa−R−S | 0.7894 | 0.7578 | 0.9217 | 0.9737 | 0.9933 | 0.8677 |

主方法相对 Magneto-Qwen 的 case-macro MRR 从 0.9501 提升到 0.9611，同时平均运行时间
约为其 13.2%。消融说明 structured matcher 与 listwise reranker 都提供增益；两者同时
移除时退化最明显。Recall@GT 仍略低于 Magneto，后续工作应优先分析候选覆盖与 Jina
重排造成的 recall/ranking 取舍，而不是继续增加未经验证的融合分支。

旧名称 `dema_jina_rerank`、`dema_fusion`、`dema_jina_no_coma`、`dema_single` 仅作为
兼容入口保留。早期自有 retriever + Jev-Single 的旧 `dema` 行为现名为 `dema_legacy`；
历史结果文件不移动、不覆盖。
