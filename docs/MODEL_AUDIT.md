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

## 当前基础版本与效果

旧版仅使用 Jev 的排序存在明显的数据集差异，因此当前基础版本改为训练-free 的
`dema_fusion`：保持 Magneto Top-20 候选，使用 Jev-Single 和 COMA++ 并行打分，再按
下面的固定原始分数公式重排：

```text
final_score = 0.4 × jev_score + 0.6 × coma_plus_score
```

COMA++ 经过双向筛选后未输出的 pair 记为 0，Top-20 之外的 retrieval tail 不变。公式
不按 dataset 分支，也不使用训练得到的 ranker。

当前 `experiment_dev.yaml` 29 个有效 case 的 case-macro 结果：

| method | MRR | Recall@GT | Recall@20 | NDCG@10 | MAP |
|---|---:|---:|---:|---:|---:|
| Magneto-Qwen | 0.9501 | 0.8259 | 0.9683 | 0.9500 | 0.9431 |
| **DeMa-Fusion** | **0.9343** | **0.8050** | **0.9683** | **0.9401** | **0.9287** |
| COMA++ | 0.9162 | 0.7382 | 0.9000 | 0.9089 | 0.9037 |
| DeMa-Single | 0.8845 | 0.6778 | 0.9683 | 0.9023 | 0.8781 |

`dema_fusion` 的结果是对已保存的候选级分数进行真实固定公式复算，不是逐 case 查看
ground truth 后选择较优方法的 oracle。正式 matcher 已固化同一公式；首次正式运行前的
约 17.1 秒/case 仅为 DeMa-Single 与 COMA++ 现有平均时间之和，不能替代 runner 实测。

各数据集 MRR 显示剩余问题主要集中在 ChEMBL：

| method | ChEMBL | GDC | Magellan | OpenData | TPC-DI | WikiData |
|---|---:|---:|---:|---:|---:|---:|
| Magneto-Qwen | 1.0000 | 0.7768 | 1.0000 | 0.9887 | 0.9933 | 0.9396 |
| DeMa-Fusion | 0.8604 | 0.8133 | 1.0000 | 0.9887 | 1.0000 | 0.9458 |
| COMA++ | 0.9758 | 0.5972 | 1.0000 | 0.9840 | 1.0000 | 0.9458 |
| DeMa-Single | 0.7894 | 0.7578 | 0.9217 | 0.9737 | 0.9933 | 0.8677 |

## 下一步验证顺序

1. 用统一 runner 正式执行 `dema_fusion`，确认保存结果与离线复算完全一致并实测时间。
2. 根据 COMA++/Jev 的置信度差和排序一致性设计动态权重，重点检查 ChEMBL。
3. 单独评估只处理低置信度 Top-3/Top-5 的小型 LLM ranker，并报告调用率和延迟。
4. 保留 `dema_single`、`dema_shared`、`dema_own_retrieval` 和旧 `dema` 作为消融，
   不覆盖其已有 prediction。
