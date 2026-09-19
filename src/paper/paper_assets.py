#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：paper_assets.py
@Description:
    论文素材核对 —— 逐条检查论文《链上AI_Agent账户行为指纹识别的研究》
    需要的每一张图、每一张表，管线是否都已产出。

    ============================================================
    一、 为什么需要这个脚本
    ============================================================
    verify_outputs.py 验的是「管线自身的产物是否合理」，
    本脚本验的是另一个问题：**论文要用的东西齐不齐**。

    两者会漏掉不同的问题。举例: 管线可以完美跑完、7 套模型全部达标，
    但论文表 5.1（超参数逐组扫描）没有对应产出 —— verify 不会报，
    等到写论文时才发现，那时重跑一轮是十几个小时。

    ============================================================
    二、 输出
    ============================================================
      figures/PAPER_ASSETS_{chain}.md   图表 → 产物 的对照清单
      退出码 0 = 齐全（可能有 warn）；1 = 有缺失

    ⚠️ 论文里的图片是 base64 内嵌的，不是按文件名引用 ——
      所以本脚本核对的是「素材是否已生成」，替换进论文仍需人工操作。
'''

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402


# (论文编号, 说明, 产物相对路径, 是否必需)
def figure_specs(tag: str) -> list:
    f = lambda n: f'figures/{n}'          # noqa: E731
    return [
        ('图4.1', '特征体系总览（57 维六族）', f('fig41_feature_system.png'), True),
        ('图4.2', '三频带与突发性定义示意', f('fig42_band_burst_def.png'), True),
        ('图4.3', '特征缺失率（协议形态导致的可得性差异）',
         f('fig43_missingness.png'), True),
        ('图5.1', '模型架构 / 五阶段管线', f('fig_model_arch.png'), True),
        ('图5.2', '消融实验框架', f('fig_ablation_frame.png'), True),
        ('图5.3', '延迟分布', f(f'fig1_latency_distribution_{tag}.png'), True),
        ('图5.4', '最快响应 Δt(p05) 的经验累积分布',
         f(f'fig2_fast_response_ecdf_{tag}.png'), True),
        ('图5.5', '三频带占比', f(f'fig3_band_shares_{tag}.png'), True),
        ('图5.6', '突发性与熵', f(f'fig4_burstiness_entropy_{tag}.png'), True),
        ('图5.7', '学习曲线（样本量收益）', f(f'fig7_learning_curve_{tag}.png'), True),
        ('图5.8', '纯行为特征集的混淆矩阵', f(f'fig5_confusion_{tag}.png'), True),
        ('图5.8b', '（附）behavior 集混淆矩阵',
         f(f'fig5_confusion_behavior_{tag}.png'), False),
        ('图5.9', '特征重要性（XGBoost 增益）', f(f'fig6_importance_{tag}.png'), True),
        ('图5.9b', '（附）增益版重要性', f(f'fig6_importance_gain_{tag}.png'), False),
        ('图5.10', 'OOF 预测概率分布', f(f'fig9_oof_proba_{tag}.png'), True),
        ('图5.11', 'One-vs-Rest ROC 曲线', f(f'fig10_roc_ovr_{tag}.png'), True),
        ('图5.12', '单棵树结构展开', f(f'fig8_tree_structure_{tag}.png'), True),
    ]


def table_specs(tag: str) -> list:
    """(编号, 说明, 产物文件, 该文件内必须出现的锚点字符串)"""
    return [
        ('表3.1', '三组真值来源与识别方法汇总',
         f'figures/SAMPLE_FUNNEL_{tag}.md', '表A 发现层'),
        ('表3.2', '活动流采集结果与有效样本量',
         f'figures/SAMPLE_FUNNEL_{tag}.md', '表B 采集与装配层'),
        ('表5.1', '超参数逐组扫描过程',
         f'figures/TRAIN_REPORT_{tag}.md', '超参数逐组扫描'),
        ('表5.2', '七组特征集定义与结果',
         f'figures/TRAIN_REPORT_{tag}.md', '套特征集对照'),
        ('表5.3', '最快响应特征的两两分布检验',
         f'figures/GONOGO_VERDICT_{tag}.md', '分布差异检验'),
        ('表5.4', '七组特征集的袋外三分类性能',
         f'figures/TRAIN_REPORT_{tag}.md', 'OOF macro-F1'),
        ('表5.5', '二分类对照实验（失效方向）',
         f'figures/TRAIN_REPORT_{tag}.md', '二分类对照实验'),
        ('表5.6', '捷径单独成模的判别力（破「Δ≈0 ⇒ 无贡献」）',
         f'figures/ROBUSTNESS_{tag}.md', '表 A'),
        ('表5.7', '二分类失效形态的两组稳健性对照',
         f'figures/ROBUSTNESS_{tag}.md', '表 B'),
    ]


MIN_FIG_BYTES = 25_000       # 空白图约 10~20KB


def main() -> int:
    ap = argparse.ArgumentParser(description='核对论文所需图表是否齐全')
    ap.add_argument('--chain', default=config.PRIMARY_CHAIN)
    args = ap.parse_args()
    tag = args.chain
    root = config.ROOT

    print('=' * 66)
    print(f'  论文素材核对 —— 《链上AI_Agent账户行为指纹识别的研究》[{tag}]')
    print('=' * 66)

    lines = [f'# 论文素材对照清单（{tag}）', '',
             '> 由 `src/paper/paper_assets.py` 自动生成。',
             '> 逐条核对论文需要的每张图、每张表是否已由管线产出。', '',
             '⚠️ 论文里的图片是 base64 内嵌的（不按文件名引用），',
             '因此本清单确认的是「素材已生成」，替换进论文仍需人工操作。', '',
             '## 图', '',
             '| 论文编号 | 内容 | 产物 | 状态 |', '|---|---|---|---|']
    missing, warns = [], []

    for num, desc, rel, required in figure_specs(tag):
        p = root / rel
        if not p.exists():
            st = '❌ 缺失'
            (missing if required else warns).append(f'{num} {rel}')
        elif p.stat().st_size < MIN_FIG_BYTES:
            st = f'⚠️ 仅 {p.stat().st_size // 1024}KB（疑似空白图）'
            warns.append(f'{num} {rel} 过小')
        else:
            st = f'✅ {p.stat().st_size // 1024}KB'
        print(f'  {st:<22} {num:<7} {desc}')
        lines.append(f'| {num} | {desc} | `{rel}` | {st} |')

    lines += ['', '## 表', '', '| 论文编号 | 内容 | 产物 | 状态 |', '|---|---|---|---|']
    for num, desc, rel, anchor in table_specs(tag):
        p = root / rel
        if not p.exists():
            st = '❌ 产出文件缺失'
            missing.append(f'{num} {rel}')
        elif anchor not in p.read_text(encoding='utf-8', errors='ignore'):
            st = f'❌ 文件中找不到「{anchor}」'
            missing.append(f'{num} {rel}:{anchor}')
        else:
            st = '✅ 已生成'
        print(f'  {st:<22} {num:<7} {desc}')
        lines.append(f'| {num} | {desc} | `{rel}` | {st} |')

    # 样本量与主要指标速览 —— 写论文时最常查的几个数
    lines += ['', '## 正文需要的关键数字', '']
    feat = config.DATA_FEAT / f'features_{tag}.csv'
    if feat.exists():
        import pandas as pd
        df = pd.read_csv(feat)
        vc = df['group'].value_counts().to_dict()
        lines.append(f'- 有效样本: **{len(df):,}**（' +
                     ' / '.join(f'{k} {v:,}' for k, v in sorted(vc.items())) + '）')
        print(f'\n  有效样本合计 {len(df):,} —— {vc}')
    for name in ('model', 'behavior'):
        mp = config.DATA_DIR / 'model' / f'{tag}_{name}' / 'metrics.json'
        if mp.exists():
            m = json.loads(mp.read_text())
            rec = m.get('oof_recall', {})
            lines.append(
                f"- {name} 集: macro-F1 **{m['oof_macro_f1']:.4f}**"
                f"，agent 召回 {rec.get('agent', float('nan')):.3f}"
                f"，特征数 {m.get('n_features')}")
    lines.append('')

    out = config.FIG_DIR / f'PAPER_ASSETS_{tag}.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n  清单 → {out.relative_to(root)}')

    print('=' * 66)
    if missing:
        print(f'  ❌ 缺 {len(missing)} 项论文必需素材:')
        for m in missing:
            print(f'     · {m}')
        print('  → 补跑: ./bin/run_all.sh --from model')
        print('=' * 66)
        return 1
    if warns:
        print(f'  ⚠️ {len(warns)} 项可选素材缺失或偏小（不影响论文主体）')
        for w in warns:
            print(f'     · {w}')
    print('  ✅ 论文所需的全部图表均已产出。')
    print('=' * 66)
    return 0


if __name__ == '__main__':
    sys.exit(main())
