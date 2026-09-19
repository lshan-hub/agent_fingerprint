#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：regen_doc_figs.py
@Description:
    论文配图重生成 —— 2026-09-07 一致性核查后的修正版

    修正内容(与核查报告一一对应):
      fig41_feature_system.png        维数口径改为与代码一致:
                                      元特征 3 维(非 2)、gas/nonce/value 18 维(非 17)、
                                      缺失模式剔除 16 维、交互族 9 维(8 维入 behavior)
      fig42_band_burst_def.png        修复 "B→-1" 负号缺字形;10s 刻度标签避让标题
      fig5_confusion_behavior_base.png 新增:behavior 集 XGBoost 分层 5 折 OOF 混淆矩阵
                                      (数据源 data/model/base_behavior/metrics.json;
                                       此前论文误用 go/no-go 阶段 RF 全特征混淆矩阵)
      fig6_importance_gain_base.png   新增:model 集最终模型的 XGBoost 增益重要性
                                      (此前论文误用 RF 不纯度重要性)
      fig_model_arch.png              Phase 4 文案改为实际行为
                                      (调参超参全量重训;best_iteration 均值逻辑未生效)

    ⚠️ 不覆盖 fig5_confusion_base.png / fig6_importance_base.png ——
       它们是 plot_gonogo.py 的管线产物,保留原样。
'''

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.common import setup_matplotlib_cjk  # noqa: E402

setup_matplotlib_cjk()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG = ROOT / 'figures'
CHAIN = 'base'


# ============================================================
#  实测口径 —— 图里的每个数字都从数据算出来，不写死
# ============================================================
# 🔴 2026-09-13 修正：此前 fig41 的「agent 组平均缺失 82%」与 fig_model_arch 的
#    「247 个地址样本」都是硬编码的旧管线遗留值：
#      · 82% 是按 gas/nonce/value **全部 18 维**(含 3 个恒可算的元特征)算的均值，
#        与图 4.3 的 98.5%/100%/100%(只算缺失敏感成员)口径不同 ⇒ 同一篇论文
#        两张图对同一个量给出两个数。
#      · 247 是旧四层漏斗管线的有效样本数，现管线是 30,000。
#    统一改为运行时从 data/features/features_{chain}.csv 现算。
def measured() -> dict:
    """从特征宽表现算图中要标注的实测量。找不到表时回退到占位符。"""
    from src.feature.f_ensemble import FeatureEnsemble as FE
    p = ROOT / 'data' / 'features' / f'features_{CHAIN}.csv'
    if not p.exists():
        return {'n_samples': None, 'miss_lo': None, 'miss_hi': None,
                'n_miss_sens': None}
    df = pd.read_csv(p)
    # 缺失敏感成员 = clean 集里、behavior 集要剔除的 16 维中真正会缺失的那些
    # （f_value_zero_ratio 恒可算，不计入）——与 gen_missing_figs.fig_missingness 同口径
    removed = [c for c in FE.CLEAN_FEATURES if c not in FE.BEHAVIOR_FEATURES]
    rate = df.groupby('group')[removed].apply(lambda x: x.isna().mean())
    sens = [c for c in removed if (rate[c].max() - rate[c].min()) > 0.30]
    ag = rate.loc['agent', sens] * 100 if 'agent' in rate.index else None
    return {'n_samples': len(df),
            'miss_lo': None if ag is None else float(ag.min()),
            'miss_hi': None if ag is None else float(ag.max()),
            'n_miss_sens': len(sens)}


M = measured()


# ============================================================
#  ① 图 4.1 —— 57 维特征体系与剔除流向(维数与代码对齐)
# ============================================================
def fig41():
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.grid(False)

    def box(x, y, w, h, text, fc, ec, tc, fs=11, lw=1.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.6,rounding_size=1.2',
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, color=tc, linespacing=1.6)

    def arrow(x1, y1, x2, y2, color, lw=1.8, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='-|>', mutation_scale=14,
                                     color=color, lw=lw,
                                     connectionstyle=f'arc3,rad={rad}'))

    ax.text(50, 99.5, '57 维特征体系与泄漏特征的剔除流向',
            ha='center', va='top', fontsize=15)

    # 左列:四个特征族(合计 57 维;元特征横跨 gas 族与交互族,不单列)
    box(2, 80, 26, 11, '延迟族 · 18 维', '#dceaf7', '#9db8d6', '#2f6fb0')
    box(2, 63, 26, 11, '节律与突发族 · 12 维', '#ece4f6', '#c0aede', '#7d5ba6')
    box(2, 46, 26, 11, '交互结构族 · 9 维', '#e2f2e3', '#a4cfa6', '#2b7a4b')
    box(2, 22, 26, 14, 'gas · nonce · value · 18 维\n(其中 2 维为元特征)',
        '#f9e0e0', '#dba8a8', '#b03636', fs=10.5)

    # 右上:纯行为特征集(绿色保留通道走上半区)
    box(76, 62, 22, 28,
        '纯行为特征集\nbehavior · 32 维\n\n延迟 18 + 节律 6 + 交互 8\n三组均可计算\n只能以行为差异解释',
        '#e8f6e9', '#2b7a4b', '#2b7a4b', fs=10.5, lw=2.2)
    ax.text(87, 58, '→ 第五章 5.3 节 七组特征集消融',
            ha='center', va='top', fontsize=10, color='#4a4f55')

    arrow(28, 86.5, 76, 84, '#2b9e4f', rad=-0.05)
    ax.text(36, 89.5, '18/18', fontsize=10.5, color='#2b7a4b')
    arrow(28, 69.5, 76, 76, '#2b9e4f', rad=-0.05)
    ax.text(36, 72.5, '6/12', fontsize=10.5, color='#2b7a4b')
    arrow(28, 52.5, 76, 68, '#2b9e4f', rad=-0.08)
    ax.text(36, 55.5, '8/9', fontsize=10.5, color='#2b7a4b')

    # 中下:三个剔除箱(合计 6 + 3 + 16 = 25 维;红色剔除通道走下半区)
    ax.text(53, 51, '两类泄漏 + 元特征: 剔除 25 维',
            ha='center', fontsize=12, color='#b03636')
    box(40, 35, 26, 11, '窗口依赖 6 维\n(绝对跨度 3 + 周期覆盖 3)',
        '#fdecec', '#c0392b', '#b03636', fs=10)
    box(40, 19, 26, 11, '协议形态元特征 3 维\n(gas 族 2 + 交互族 1)',
        '#fdecec', '#c0392b', '#b03636', fs=10)
    # 口径与图 4.3 完全一致：只报缺失敏感成员对 agent 组的缺失率区间
    miss_txt = ('(agent 组缺失率区间未知)' if M['miss_lo'] is None else
                f"(其中 {M['n_miss_sens']} 维缺失敏感,\n"
                f"对 agent 组缺失 {M['miss_lo']:.1f}%~{M['miss_hi']:.1f}%)")
    box(40, 3, 26, 11, f'缺失模式 16 维\n{miss_txt}',
        '#fdecec', '#c0392b', '#b03636', fs=9)

    arrow(24, 63, 40, 42, '#c0392b', lw=1.5, rad=0.12)      # 节律 → 窗口依赖 6
    arrow(28, 48, 40, 27.5, '#c0392b', lw=1.5, rad=0.10)    # 交互 → 元特征 1
    arrow(28, 31, 40, 25.5, '#c0392b', lw=1.5, rad=0.02)    # gnv  → 元特征 2
    arrow(28, 26, 40, 9.5, '#c0392b', lw=1.5, rad=0.08)     # gnv  → 缺失模式 16

    fig.savefig(FIG / 'fig41_feature_system.png')
    plt.close(fig)
    print('✓ fig41_feature_system.png')


# ============================================================
#  ② 图 4.2 —— 三频带定义 + 突发性三形态(修负号/标签避让)
# ============================================================
def fig42():
    rng = np.random.default_rng(42)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.0, 6.4), height_ratios=[1.0, 1.35],
        gridspec_kw=dict(hspace=0.55))

    # --- (a) 三频带 ---
    ax1.set_xscale('log')
    ax1.set_xlim(0.3, 320)
    ax1.set_ylim(0, 1)
    ax1.axvspan(0.3, 2, color='#f6dfdf')
    ax1.axvspan(2, 10, color='#dfe9f6')
    ax1.axvspan(10, 320, color='#dff0e2')
    for x in (2, 10):
        ax1.axvline(x, ls='--', color='#666', lw=1.2)
        ax1.text(x, 1.04, f'{x}s', ha='center', va='bottom',
                 fontsize=10.5, color='#555')
    ax1.text(0.78, 0.5, '机器带 Δt<2s\n贴块脚本响应', ha='center',
             va='center', fontsize=11, color='#b03636', linespacing=1.8)
    ax1.text(4.5, 0.5, '边界带 2~10s\n出块+链下推理\n(agent 预期主频带)',
             ha='center', va='center', fontsize=11, color='#2f6fb0',
             linespacing=1.8)
    ax1.text(57, 0.5, '人类带 Δt≥10s\n人工操作节奏', ha='center',
             va='center', fontsize=11, color='#2b7a4b', linespacing=1.8)
    ax1.set_yticks([])
    ax1.set_xticks([0.5, 1, 2, 5, 10, 30, 60, 300])
    ax1.set_xticklabels(['0.5s', '1s', '2s', '5s', '10s', '30s', '1m', '5m'])
    ax1.set_xlabel('相邻活动间隔 Δt (对数轴)')
    ax1.set_title('(a) 三频带的定义(式 4.2)', pad=22)
    ax1.grid(False)

    # --- (b) 突发性三形态 ---
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.6, 2.9)
    rows = [
        (2, '#2f6fb0', 'B → 1   事件驱动/人类: 成簇突发, 簇间长静默'),
        (1, '#7a7f87', 'B ≈ 0   泊松过程: 无记忆随机'),
        (0, '#c0392b', 'B → -1  定时任务: 严格周期'),
    ]
    # 成簇: 6 个簇,簇内 3~6 个紧密事件
    burst = np.concatenate([c + rng.uniform(0, 3.2, rng.integers(3, 7))
                            for c in [4, 21, 27, 52, 74, 96]])
    poisson = np.cumsum(rng.exponential(2.6, 60))
    poisson = poisson[poisson < 99]
    periodic = np.arange(2, 99, 3.4)
    for (y, color, label), xs in zip(rows, [burst, poisson, periodic]):
        ax2.vlines(xs, y - 0.16, y + 0.16, color=color, lw=1.6)
        ax2.hlines(y, 0, 100, color='#d5d8dc', lw=0.8, zorder=0)
        ax2.text(0, y + 0.42, label, fontsize=11, color='#3a3f45')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlabel('时间 →')
    ax2.set_title('(b) 突发性系数 B 的三种形态(式 4.3)', pad=10)
    ax2.grid(False)
    for s in ('left', 'bottom'):
        ax2.spines[s].set_visible(False)

    fig.savefig(FIG / 'fig42_band_burst_def.png')
    plt.close(fig)
    print('✓ fig42_band_burst_def.png')


# ============================================================
#  ③ 图 5.8 —— behavior 集 XGBoost 分层 5 折 OOF 混淆矩阵
# ============================================================
def fig5_confusion_behavior():
    m = json.loads((ROOT / 'data/model/base_behavior/metrics.json').read_text())
    cm = np.array(m['confusion_matrix'], dtype=float)   # 行/列序 = m['labels']
    labels = m['labels']                                 # ['human','bot','agent']
    order = ['bot', 'agent', 'human']                    # 展示序与 go/no-go 图一致
    idx = [labels.index(g) for g in order]
    cm = cm[np.ix_(idx, idx)]
    names = {'bot': '传统脚本 Bot (MEV)', 'agent': 'AI Agent (Olas/x402)',
             'human': '人类 (CEX 提现)'}

    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([names[g] for g in order], rotation=18,
                       ha='right', fontsize=9)
    ax.set_yticklabels([names[g] for g in order], fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cmn[i, j]:.2f}\n({int(cm[i, j])})',
                    ha='center', va='center',
                    color='white' if cmn[i, j] > 0.5 else '#1a1d21',
                    fontsize=10)
    ax.set_xlabel('预测')
    ax.set_ylabel('真实')
    ax.set_title('纯行为特征集(behavior)三分类混淆矩阵\n'
                 f"(XGBoost, 分层 5 折袋外, macro-F1 = "
                 f"{m['oof_macro_f1']:.4f})", pad=12)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.savefig(FIG / 'fig5_confusion_behavior_base.png')
    plt.close(fig)
    print('✓ fig5_confusion_behavior_base.png')


# ============================================================
#  ④ 图 5.9 —— model 集最终模型的 XGBoost 增益重要性
# ============================================================
def fig6_importance_gain(top: int = 18):
    from src.feature.f_ensemble import LATENCY_ONLY_FEATURES
    with open(ROOT / 'data/model/base_model/xgb_clf_full.pkl', 'rb') as f:
        model = pickle.load(f)
    gain = model.get_booster().get_score(importance_type='gain')
    s = pd.Series(gain).sort_values(ascending=True).tail(top)

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.32 * len(s))))
    is_lat = [f in LATENCY_ONLY_FEATURES for f in s.index]
    ax.barh(range(len(s)), s.values,
            color=['#2f6fb0' if b else '#c9ced6' for b in is_lat])
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=9)
    ax.set_xlabel('XGBoost 增益重要性(model 集最终模型)')
    ax.set_title('特征重要性(增益;蓝色 = 延迟族)\n'
                 'gas 可得率 / nonce 跳变等结构性捷径特征占据前列', pad=12)
    ax.grid(axis='y', visible=False)
    fig.savefig(FIG / 'fig6_importance_gain_base.png')
    plt.close(fig)
    print('✓ fig6_importance_gain_base.png')


# ============================================================
#  ⑤ 图 5.1 —— 模型总体架构(Phase 4 文案与实际行为对齐)
# ============================================================
def fig_model_arch():
    fig, ax = plt.subplots(figsize=(14.6, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.grid(False)

    def box(x, y, w, h, title, body, fc, ec, tfc='white', bfc=None, fs=11.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.5,rounding_size=1.0',
                                    fc='white', ec=ec, lw=1.2))
        ax.add_patch(FancyBboxPatch((x, y + h * 0.62), w, h * 0.38,
                                    boxstyle='round,pad=0.5,rounding_size=1.0',
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h * 0.81, title, ha='center', va='center',
                fontsize=fs, color=tfc)
        ax.text(x + w / 2, y + h * 0.30, body, ha='center', va='center',
                fontsize=fs - 2, color=bfc or '#3a3f45', linespacing=1.7)

    def flat(x, y, w, h, text, fc, ec, tc='white', fs=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.5,rounding_size=1.0',
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, color=tc, linespacing=1.7)

    def arr(x1, y1, x2, y2, ls='-'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='-|>', mutation_scale=13,
                                     color='#6a6f77', lw=1.5, linestyle=ls))

    n_txt = ('N 个地址样本' if M['n_samples'] is None
             else f"{M['n_samples']:,} 个地址样本")
    flat(18, 90, 64, 8,
         f'特征宽表 · {n_txt} × 57 维行为特征(含 group 标签)',
         '#2a9d8f', '#1f776d')

    phases = [
        (1,  'Phase 0 数据校验', '地址去重·防泄漏\n剔除规模黑名单', '#e8a23d', '#b57718'),
        (21, 'Phase 1 特征选择', '常数/全NaN → 相关去重\n|ρ|>0.95 → 零增益\n→ Top-40 上限', '#e07a3f', '#a85423'),
        (41, 'Phase 2 网格搜索', '4 轮分组扫描\n3 折 CV · macro-F1 准则', '#69a05c', '#47703e'),
        (61, 'Phase 3 分层5折OOF', '折外概率预测\n折内无标签泄漏', '#3f6fb5', '#2b4d80'),
        (81, 'Phase 4 全量重训', '以调参超参全量重训\n落盘线上模型', '#7c5fb0', '#57407e'),
    ]
    for i, (x, t, b, fc, ec) in enumerate(phases):
        box(x, 66, 18, 18, t, b, fc, ec)
        if i:
            arr(phases[i - 1][0] + 18.4, 75, x - 0.4, 75)

    arr(50, 90, 50, 84.6)
    arr(50, 66, 50, 60.6)

    flat(18, 50, 64, 10,
         'XGBoost 多分类器(multi:softprob)\n加法树模型 · 二阶梯度增益分裂 · 缺失值默认方向',
         '#6d55a3', '#4d3a77')

    for x, t in ((22, 'P(human)'), (44, 'P(bot)'), (66, 'P(agent)')):
        flat(x, 36, 12, 7, t, {'P(human)': '#c26296', 'P(bot)': '#c94f5e',
                               'P(agent)': '#3d8f8a'}[t], '#5a5f66')
        arr(x + 6, 50, x + 6, 43.4)
        arr(x + 6, 36, 44, 27.6)

    flat(38, 20, 12, 7, 'argmax', '#8a8f96', '#6a6f77')
    flat(56, 20, 20, 7, '三分类判别结果', '#c9982e', '#997117')
    arr(50.4, 23.5, 55.6, 23.5)

    box(80, 14, 17, 15, 'Phase 5 二分类对照',
        'human/bot 训练\n→ 判 agent 去向\n(按特征集分别运行)', '#7a7f87', '#5a5f66')
    ax.add_patch(FancyArrowPatch((76.4, 23.5), (79.6, 23.5),
                                 arrowstyle='-|>', mutation_scale=13,
                                 color='#6a6f77', lw=1.5, linestyle='--'))

    fig.savefig(FIG / 'fig_model_arch.png')
    plt.close(fig)
    print('✓ fig_model_arch.png')


if __name__ == '__main__':
    fig41()
    fig42()
    fig5_confusion_behavior()
    fig6_importance_gain()
    fig_model_arch()
    print('全部完成 →', FIG)
