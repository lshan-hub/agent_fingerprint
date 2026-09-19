# 论文素材对照清单（base）

> 由 `src/paper/paper_assets.py` 自动生成。
> 逐条核对论文需要的每张图、每张表是否已由管线产出。

⚠️ 论文里的图片是 base64 内嵌的（不按文件名引用），
因此本清单确认的是「素材已生成」，替换进论文仍需人工操作。

## 图

| 论文编号 | 内容 | 产物 | 状态 |
|---|---|---|---|
| 图4.1 | 特征体系总览（57 维六族） | `figures/fig41_feature_system.png` | ✅ 171KB |
| 图4.2 | 三频带与突发性定义示意 | `figures/fig42_band_burst_def.png` | ✅ 101KB |
| 图4.3 | 特征缺失率（协议形态导致的可得性差异） | `figures/fig43_missingness.png` | ✅ 56KB |
| 图5.1 | 模型架构 / 五阶段管线 | `figures/fig_model_arch.png` | ✅ 191KB |
| 图5.2 | 消融实验框架 | `figures/fig_ablation_frame.png` | ✅ 115KB |
| 图5.3 | 延迟分布 | `figures/fig1_latency_distribution_base.png` | ✅ 108KB |
| 图5.4 | 最快响应 Δt(p05) 的经验累积分布 | `figures/fig2_fast_response_ecdf_base.png` | ✅ 90KB |
| 图5.5 | 三频带占比 | `figures/fig3_band_shares_base.png` | ✅ 71KB |
| 图5.6 | 突发性与熵 | `figures/fig4_burstiness_entropy_base.png` | ✅ 643KB |
| 图5.7 | 学习曲线（样本量收益） | `figures/fig7_learning_curve_base.png` | ✅ 137KB |
| 图5.8 | 纯行为特征集的混淆矩阵 | `figures/fig5_confusion_base.png` | ✅ 83KB |
| 图5.8b | （附）behavior 集混淆矩阵 | `figures/fig5_confusion_behavior_base.png` | ✅ 95KB |
| 图5.9 | 特征重要性（XGBoost 增益） | `figures/fig6_importance_base.png` | ✅ 89KB |
| 图5.9b | （附）增益版重要性 | `figures/fig6_importance_gain_base.png` | ✅ 91KB |
| 图5.10 | OOF 预测概率分布 | `figures/fig9_oof_proba_base.png` | ✅ 62KB |
| 图5.11 | One-vs-Rest ROC 曲线 | `figures/fig10_roc_ovr_base.png` | ✅ 62KB |
| 图5.12 | 单棵树结构展开 | `figures/fig8_tree_structure_base.png` | ✅ 109KB |

## 表

| 论文编号 | 内容 | 产物 | 状态 |
|---|---|---|---|
| 表3.1 | 三组真值来源与识别方法汇总 | `figures/SAMPLE_FUNNEL_base.md` | ✅ 已生成 |
| 表3.2 | 活动流采集结果与有效样本量 | `figures/SAMPLE_FUNNEL_base.md` | ✅ 已生成 |
| 表5.1 | 超参数逐组扫描过程 | `figures/TRAIN_REPORT_base.md` | ✅ 已生成 |
| 表5.2 | 七组特征集定义与结果 | `figures/TRAIN_REPORT_base.md` | ✅ 已生成 |
| 表5.3 | 最快响应特征的两两分布检验 | `figures/GONOGO_VERDICT_base.md` | ✅ 已生成 |
| 表5.4 | 七组特征集的袋外三分类性能 | `figures/TRAIN_REPORT_base.md` | ✅ 已生成 |
| 表5.5 | 二分类对照实验（失效方向） | `figures/TRAIN_REPORT_base.md` | ✅ 已生成 |
| 表5.6 | 捷径单独成模的判别力（破「Δ≈0 ⇒ 无贡献」） | `figures/ROBUSTNESS_base.md` | ✅ 已生成 |
| 表5.7 | 二分类失效形态的两组稳健性对照 | `figures/ROBUSTNESS_base.md` | ✅ 已生成 |

## 正文需要的关键数字

- 有效样本: **30,000**（agent 18,305 / bot 5,695 / human 6,000）
- model 集: macro-F1 **0.9997**，agent 召回 1.000，特征数 32
- behavior 集: macro-F1 **0.9812**，agent 召回 0.988，特征数 21
