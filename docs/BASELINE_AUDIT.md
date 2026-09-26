# Baseline implementation audit

审计日期：2026-09-25。审计目标不是声称所有方法“名字一样就等价”，而是明确每个实现的来源、固定版本和差异。

## 参考版本

| 项目 | 固定提交 | 用途 |
|---|---|---|
| [magneto-matcher](https://github.com/VIDA-NYU/magneto-matcher) | `6620623265fc7feac0f053996e62b68a13a72a57` | Magneto 本体 |
| [data-harmonization-benchmark](https://github.com/VIDA-NYU/data-harmonization-benchmark) | `3207e37e898af06211ef1a47316f9c07dc3572cd` | COMA/COMAInst/ISResMat 等实验 wrapper |
| [Valentine](https://github.com/delftdata/valentine) | `5d5163f04da304985bd51a476ccf7653de3979c3` | 传统 matcher API |
| [ISResMat](https://github.com/duxyad/ISResMat) | `7db84986ac87aac84f2296a6d6e09550dd0f420e` | ISResMat 参考实现 |
| [Unicorn](https://github.com/ruc-datalab/Unicorn) | `5424e585f6c739db3c1ca610e0db079392dfbfd6` | Unicorn 参考实现 |

## 逐方法结论

| 方法 | 一致性 | 结论 |
|---|---|---|
| `magneto_qwen` | 上游源码 + 明示补丁 | `vendor/magneto` 与固定提交一致；只将 `litellm.completion` 换为本地 endpoint，并增加请求统计。完整 diff 见 `vendor/magneto/PATCHES.md`。模型从 GPT-4o-mini 改为 Qwen3.5-9B，因此这是受控变体，不是原论文数值复现。 |
| `coma` | 参数对齐、运行时不同 | wrapper 的 `max_n=20`、schema-only，以及 COMA 默认 `delta=0.15` 已对齐。当前 Valentine 1.x 使用 Python 实现；参考 benchmark 曾依赖 Java COMA，因此不能宣称逐分数相同。 |
| `coma_plus` | 参数对齐、采样可复现 | 对齐 `ComaInst`：每张表最多随机采样 500 行、instance matcher 开启、`max_n=20`。本仓库固定 `sample_seed=42`；上游 wrapper 未固定 seed。 |
| `distribution` | 当前 Valentine 默认参数 | `threshold1=0.15`、`threshold2=0.15`、`quantiles=256`、单进程，与当前实现默认值一致。 |
| `similarity_flooding` | 当前 Valentine 默认参数 | inverse-average、formula C、prefix/suffix string matcher。 |
| `unicorn` | 推理重实现 | 使用发布的 UnicornPlus checkpoint、DeBERTa-base、6 experts、`[ATT]/[VAL]` 序列化、前 20 个 unique values、128 tokens 和 match logit。不是直接执行上游仓库，升级依赖时必须跑慢测试比对 state dict。 |
| `isresmat` | 近似重实现 | 主要超参数已按 wrapper 对齐，但缺少参考实现的完整验证/早停和部分数据增强行为。结果必须标注为 reimplementation，不能当作官方数值复现。 |

## 排名适配

统一 evaluator 要求每个 source column 对所有 target columns 给出完整排序。原始 baseline 只返回部分 pair 时，`dema.model.ranking.ranking_from_scores` 把未返回 pair 放在已评分 pair 之后，并按 target schema 顺序稳定打破平局。这个适配不会向 matcher 暴露 ground truth，但与只评估 top-k 的原始脚本不是完全相同的输出协议。

## 维护规则

1. `vendor/magneto` 的任何功能改动都必须写入 `PATCHES.md`。
2. 更新 Valentine、checkpoint 或参考提交后，必须更新本文件并重跑 baseline tests。
3. 报告实验结果时区分 `upstream code`、`ported wrapper`、`reimplementation`，不要统一写成“official implementation”。
