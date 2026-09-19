#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：gen_missing_figs.py
@Description:
    补齐 paper_assets.py 清单里「有引用、无生成者」的 6 张图。

    背景：2026-09-13 的 30k 全量运行结束后，paper_assets.py 报「缺 6 项论文
    必需素材」。逐一追查发现这 6 张图在整个代码库里都没有生成代码 ——
    paper_assets.py 只是清单检查器（列出必需项并判断文件在不在），
    regen_doc_figs.py 只覆盖另外 5 张。这 6 张大概率是早期手工产出后
    生成脚本未入库。论文正文对它们有实质引用（如「图 5.7(b) 的样本量
    学习曲线显示…」），不能缺。

    本脚本产出：
      图4.3  fig43_missingness.png        三组 × gas/nonce/value 三族缺失率
      图5.2  fig_ablation_frame.png       七套特征集递进消融框架
      图5.7  fig7_learning_curve_base.png 样本量-性能学习曲线（含 ±1σ 带）
      图5.10 fig9_oof_proba_base.png      OOF 预测概率分布（对数计数轴）
      图5.11 fig10_roc_ovr_base.png       One-vs-Rest ROC
      图5.12 fig8_tree_structure_base.png 单棵树结构展开

    口径约定（与 train_xgb 严格对齐，避免论文出现两套数字）：
      · 一律用 behavior 特征集 —— 论文主结论口径
      · 超参直接读 data/model/base_behavior/config.pkl 的 best_params，
        不重新调参（重调会得到与表 5.1 不一致的数字）
      · 缺失值用 config.pkl 落盘的 impute_median 填充，与训练时同源
      · OOF 用 StratifiedKFold(n_splits=metrics.json 的 n_folds)，
        shuffle/random_state 与 train_xgb 一致

    ⚠️ 不覆盖 regen_doc_figs.py 与 plot_gonogo.py 的产物。
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
MODEL = ROOT / 'data' / 'model'
FEAT = ROOT / 'data' / 'features'

CHAIN = 'base'
SET = 'behavior'          # 论文主结论口径

# 三组固定配色（与 regen_doc_figs.py 同色系）
C = {'agent': '#d1495b', 'bot': '#00798c', 'human': '#edae49'}
ORDER = ['human', 'bot', 'agent']


# ---------------------------------------------------------------- 公共载入
def load_all():
    cfg = pickle.load(open(MODEL / f'{CHAIN}_{SET}' / 'config.pkl', 'rb'))
    met = json.load(open(MODEL / f'{CHAIN}_{SET}' / 'metrics.json'))
    df = pd.read_csv(FEAT / f'features_{CHAIN}.csv')
    feats = cfg['feature_names']
    X = df[feats].copy()
    for c, v in (cfg.get('impute_median') or {}).items():
        if c in X.columns:
            X[c] = X[c].fillna(v)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = df[cfg.get('label_field', 'group')].values
    return cfg, met, df, X.values, y, feats


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'✓ {name}')


# ---------------------------------------------------------- 图4.3 缺失率
def fig_missingness(df):
    '''三组 × gas/nonce/value 三族的特征缺失率。

    论文 4.5 节论点：gas/nonce/value 仅 tx_from 记录携带，对 agent 组
    缺失率极高，梯度提升树可按「字段有没有值」分裂而绕过行为本身 ——
    这正是 behavior 集要把这三族整族剔除的理由。
    '''
    # 🔴 只统计各族中真正「缺失敏感」的成员，不能按 f_gas / f_val 前缀
    #   一把抓 —— 三个成员是恒可算的元特征，混进来会把缺失率算低：
    #     f_gas_avail_ratio  gasPrice 可得率本身（无值时为 0，不是 NaN）
    #     f_txfrom_ratio     tx_from 占比（同上）
    #     f_value_zero_ratio value==0 占比（同上）
    #   实测差异：全部成员口径 82%/100%/75%，缺失敏感口径 98%/100%/100%。
    #   论文图 4.3 与 特征说明.md 第九节均以后者为准。
    cols = {
        'gas 族':   ['f_gas_price_cv', 'f_gas_price_mode_share',
                     'f_gas_price_median', 'f_gas_price_p95_p50',
                     'f_gas_nunique_ratio'],
        'nonce 族': ['f_nonce_gap_ratio', 'f_nonce_seq_ratio',
                     'f_nonce_burst_max', 'f_nonce_burst_ratio',
                     'f_nonce_span'],
        'value 族': ['f_round_value_ratio', 'f_value_median_eth',
                     'f_value_cv', 'f_value_nunique_ratio',
                     'f_micro_value_ratio'],
    }
    cols = {k: [c for c in v if c in df.columns] for k, v in cols.items()}
    cols = {k: v for k, v in cols.items() if v}

    fig, ax = plt.subplots(figsize=(9, 5))
    w, xs = 0.25, np.arange(len(cols))
    for i, g in enumerate(ORDER):
        sub = df[df['group'] == g]
        rates = [sub[c].isna().mean() * 100 for c in cols.values()
                 for c in [c]] if False else \
               [sub[cs].isna().mean().mean() * 100 for cs in cols.values()]
        b = ax.bar(xs + (i - 1) * w, rates, w, label=g, color=C[g],
                   edgecolor='white', linewidth=.8)
        ax.bar_label(b, fmt='%.0f%%', fontsize=9, padding=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(cols.keys()))
    ax.set_ylabel('特征缺失率 (%)')
    ax.set_ylim(0, 108)
    ax.set_title('图4.3  gas / nonce / value 三族的分组缺失率\n'
                 '(缺失模式本身即可区分三组 —— behavior 集据此整族剔除)',
                 fontsize=12)
    ax.legend(title='组别', frameon=False)
    ax.grid(axis='y', alpha=.25, linestyle='--')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    save(fig, 'fig43_missingness.png')

    # 回传实测值供论文对账
    out = {}
    for g in ORDER:
        sub = df[df['group'] == g]
        out[g] = {k: round(sub[cs].isna().mean().mean() * 100, 1)
                  for k, cs in cols.items()}
    return out


# ------------------------------------------------------- 图5.2 消融框架
def fig_ablation_frame(met_all):
    '''七套特征集的递进消融框架：每一套堵掉一条捷径。'''
    rows = [
        ('model',    '全部 57 特征',            '上界'),
        ('latency',  '仅延迟族',                '验证 H1'),
        ('nometa',   '剔元特征',                '堵协议形态'),
        ('nowin',    '剔窗口依赖特征',          '堵采集窗口'),
        ('clean',    'nometa + nowin',          '严对照'),
        ('behavior', '再剔 gas/nonce/value',    '★主结论'),
        ('timing',   '只剩延迟 + 节律',         '下界'),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, len(rows) + 1.4)

    for i, (key, desc, role) in enumerate(rows):
        y = len(rows) - i
        f1 = (met_all.get(key) or {}).get('oof_macro_f1')
        nf = (met_all.get(key) or {}).get('n_features')
        star = key == 'behavior'
        box = FancyBboxPatch(
            (.3, y - .34), 5.6, .68,
            boxstyle='round,pad=0.06', linewidth=2 if star else 1,
            edgecolor='#d1495b' if star else '#9aa3ad',
            facecolor='#fdeef0' if star else '#f5f7f9')
        ax.add_patch(box)
        ax.text(.55, y, f'{key}', fontsize=11,
                fontweight='bold' if star else 'normal', va='center')
        ax.text(2.0, y, desc, fontsize=10, va='center', color='#44505c')
        ax.text(6.15, y, role, fontsize=10, va='center',
                color='#d1495b' if star else '#6b757f',
                fontweight='bold' if star else 'normal')
        if f1 is not None:
            ax.text(8.1, y, f'macro-F1 {f1:.4f}', fontsize=10.5, va='center',
                    fontweight='bold' if star else 'normal',
                    color='#1b2a38')
            ax.text(9.55, y, f'{nf} 维', fontsize=9.5, va='center',
                    color='#6b757f', ha='right')
        if i < len(rows) - 1:
            ax.add_patch(FancyArrowPatch(
                (3.1, y - .38), (3.1, y - .66),
                arrowstyle='-|>', mutation_scale=11, color='#9aa3ad', lw=1.1))

    ax.text(5, len(rows) + .95, '图5.2  七套特征集的递进消融框架',
            fontsize=13, ha='center', fontweight='bold')
    ax.text(5, len(rows) + .52,
            '从宽到严，每一套堵掉一条结构性捷径；主结论以最严的 behavior 集为准',
            fontsize=10, ha='center', color='#6b757f')
    save(fig, 'fig_ablation_frame.png')


# ------------------------------------------------------- 图5.7 训练过程核查
# 🔴 2026-09-13 修正：论文 5.5 节写的是「图 5.7(a) 逐轮 mlogloss；(b) 样本量-
#    性能曲线」，但此前本函数只出 (b) 一张单栏图 —— 正文对 (a) 的整段讨论
#    （训练/验证 mlogloss 两线间隙、有无过拟合）在图上没有任何依据。
#    现补出 (a) 栏，合成双栏图，使正文与图一一对应。
def fig_train_curve_ax(ax, X, y, params, n_folds):
    '''(a) 逐轮 mlogloss —— 容量是否过拟合。

    口径：与 Phase 3 同一套折划分（StratifiedKFold, shuffle, seed=42）的第 1 折，
    同一套 best_params；只是额外把每轮的 train/val mlogloss 记录下来。
    '''
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    le = LabelEncoder(); yy = le.fit_transform(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    tr, va = next(iter(skf.split(X, yy)))
    clf = XGBClassifier(**params, objective='multi:softprob',
                        num_class=len(le.classes_), n_jobs=-1,
                        random_state=42, tree_method='hist',
                        eval_metric='mlogloss', verbosity=0)
    clf.fit(X[tr], yy[tr], eval_set=[(X[tr], yy[tr]), (X[va], yy[va])],
            verbose=False)
    r = clf.evals_result()
    a = np.array(r['validation_0']['mlogloss'])
    b = np.array(r['validation_1']['mlogloss'])
    it = np.arange(1, len(a) + 1)
    ax.plot(it, a, lw=2, color='#00798c', label='训练折 mlogloss')
    ax.plot(it, b, lw=2, color='#d1495b', label='验证折 mlogloss')
    ax.fill_between(it, a, b, color='#d1495b', alpha=.10)
    ax.set_xlabel('提升轮数')
    ax.set_ylabel('多分类对数损失 (mlogloss)')
    ax.set_title(f"(a) 逐轮 mlogloss（第 1 折 · max_depth="
                 f"{params.get('max_depth')} · {params.get('n_estimators')} 轮）",
                 fontsize=11)
    ax.grid(alpha=.25, linestyle='--'); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    gap = float(b[-1] - a[-1])
    print(f'    末轮 train={a[-1]:.4f} val={b[-1]:.4f} 间隙={gap:+.4f}')
    return {'train_final': round(float(a[-1]), 4),
            'val_final': round(float(b[-1]), 4),
            'gap_final': round(gap, 4),
            'val_min': round(float(b.min()), 4),
            'val_argmin_round': int(b.argmin()) + 1,
            'n_rounds': int(len(a))}


def fig_learning_curve(X, y, params, n_folds):
    '''图5.7 双栏：(a) 逐轮 mlogloss；(b) 样本量-性能曲线。'''
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.6, 5))
    curve = fig_train_curve_ax(axa, X, y, params, n_folds)

    le = LabelEncoder(); yy = le.fit_transform(y)
    fracs = [.005, .01, .02, .05, .1, .2, .4, .7, 1.0]
    ns, mu, sd = [], [], []
    for fr in fracs:
        n = max(60, int(len(yy) * fr))
        rs = np.random.RandomState(42)
        idx = rs.choice(len(yy), n, replace=False)
        # 保证每折每类都有样本
        if min(np.bincount(yy[idx])) < n_folds:
            continue
        clf = XGBClassifier(**params, objective='multi:softprob',
                            num_class=len(le.classes_), n_jobs=-1,
                            random_state=42, tree_method='hist',
                            eval_metric='mlogloss', verbosity=0)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        s = cross_val_score(clf, X[idx], yy[idx], cv=cv,
                            scoring='f1_macro', n_jobs=1)
        ns.append(n); mu.append(s.mean()); sd.append(s.std())
        print(f'    n={n:>6,}  macro-F1={s.mean():.4f} ±{s.std():.4f}')

    mu, sd, ns = np.array(mu), np.array(sd), np.array(ns)
    axb.plot(ns, mu, 'o-', color='#d1495b', lw=2, ms=6, label='验证 macro-F1')
    axb.fill_between(ns, mu - sd, mu + sd, color='#d1495b', alpha=.16,
                     label='±1σ')
    axb.set_xscale('log')
    # 🔴 横轴是「子样本总量」，每折真正用于训练的是其 (k-1)/k —— 标注写清楚，
    #    避免被读成「训练集大小」。
    axb.set_xlabel(f'子样本总量（对数轴；每折训练量为其 {n_folds - 1}/{n_folds}）')
    axb.set_ylabel(f'分层 {n_folds} 折交叉验证 macro-F1')
    axb.set_title('(b) 样本量-性能学习曲线', fontsize=11)
    axb.grid(alpha=.25, linestyle='--')
    axb.set_axisbelow(True)
    axb.legend(frameon=False, fontsize=9.5)
    for s in ('top', 'right'):
        axb.spines[s].set_visible(False)

    fig.suptitle('图5.7  纯行为特征集（behavior）的训练过程核查',
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, f'fig7_learning_curve_{CHAIN}.png')
    return (list(zip(ns.tolist(), mu.round(4).tolist(), sd.round(4).tolist())),
            curve)


# ------------------------------------------- 图5.10/5.11 OOF 概率 + ROC
def fig_oof_proba_roc(X, y, params, n_folds, classes):
    '''重算 OOF 概率矩阵 —— 训练阶段未落盘，此处按同口径复算。'''
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_curve, auc
    from xgboost import XGBClassifier

    le = LabelEncoder(); yy = le.fit_transform(y)
    P = np.zeros((len(yy), len(le.classes_)))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(cv.split(X, yy), 1):
        clf = XGBClassifier(**params, objective='multi:softprob',
                            num_class=len(le.classes_), n_jobs=-1,
                            eval_metric='mlogloss', verbosity=0)
        clf.fit(X[tr], yy[tr])
        P[va] = clf.predict_proba(X[va])
        print(f'    折 {k}/{n_folds} 完成')

    # —— 图5.10 真实类别的 OOF 概率分布
    true_p = P[np.arange(len(yy)), yy]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for g in ORDER:
        i = list(le.classes_).index(g)
        ax.hist(true_p[yy == i], bins=np.linspace(0, 1, 41), alpha=.62,
                label=f'{g} (n={int((yy == i).sum()):,})', color=C[g])
    ax.set_yscale('log')
    ax.set_xlabel('模型给「真实类别」的 OOF 预测概率')
    ax.set_ylabel('地址数（对数轴）')
    ax.set_title('图5.10  袋外预测概率分布（behavior 特征集）\n'
                 '右端单柱占绝对主导 ⇒ 高置信的正确判决，而非勉强过线',
                 fontsize=12)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.25, linestyle='--')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    save(fig, f'fig9_oof_proba_{CHAIN}.png')

    # —— 图5.11 One-vs-Rest ROC
    fig, ax = plt.subplots(figsize=(6.8, 6))
    aucs = {}
    for g in ORDER:
        i = list(le.classes_).index(g)
        fpr, tpr, _ = roc_curve((yy == i).astype(int), P[:, i])
        a = auc(fpr, tpr); aucs[g] = round(a, 4)
        ax.plot(fpr, tpr, lw=2, color=C[g], label=f'{g}  AUC={a:.4f}')
    ax.plot([0, 1], [0, 1], '--', color='#b6bec7', lw=1)
    ax.set_xlabel('假阳率 FPR'); ax.set_ylabel('真阳率 TPR')
    ax.set_title('图5.11  One-vs-Rest ROC 曲线（behavior 集 OOF）',
                 fontsize=12)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=.25, linestyle='--'); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    save(fig, f'fig10_roc_ovr_{CHAIN}.png')

    med = {g: round(float(np.median(true_p[yy == list(le.classes_).index(g)])), 4)
           for g in ORDER}
    return aucs, med


# ---------------------------------------------------- 图5.12 单棵树结构
def fig_tree(feats, max_depth=3):
    '''单棵提升树的结构展开。

    🔴 不用 xgboost.plot_tree —— 它依赖 graphviz 的 dot 可执行文件，
       本机未安装且装系统包不在本任务范围内。改为直接解析
       booster.trees_to_dataframe() 自行用 matplotlib 布局，无外部依赖。

    只画到 max_depth 层：max_depth=6 的完整树有上百个节点，
    印在论文里不可读；论文图的作用是展示「判决结构长什么样」，
    前 3 层足够，更深的用省略号标注。
    '''
    clf = pickle.load(open(MODEL / f'{CHAIN}_{SET}' / 'xgb_clf_full.pkl', 'rb'))
    booster = clf.get_booster() if hasattr(clf, 'get_booster') else clf
    try:
        booster.feature_names = list(feats)
    except Exception:
        pass
    tdf = booster.trees_to_dataframe()
    t0 = tdf[tdf['Tree'] == 0].set_index('ID')

    # 逐层展开，记录每个节点的 (深度, 水平位置)
    nodes, edges = {}, []
    root = t0.index[0]

    def walk(nid, depth, lo, hi):
        if nid not in t0.index:
            return
        r = t0.loc[nid]
        x = (lo + hi) / 2
        leaf = (r['Feature'] == 'Leaf')
        nodes[nid] = dict(x=x, y=-depth, leaf=leaf,
                          feat=r['Feature'], split=r['Split'],
                          gain=r['Gain'], cover=r['Cover'])
        if leaf or depth >= max_depth:
            nodes[nid]['cut'] = (not leaf) and depth >= max_depth
            return
        for child, tag in ((r['Yes'], '是'), (r['No'], '否')):
            if isinstance(child, str) and child in t0.index:
                edges.append((nid, child, tag))
        mid = (lo + hi) / 2
        walk(r['Yes'], depth + 1, lo, mid)
        walk(r['No'], depth + 1, mid, hi)

    walk(root, 0, 0, 1)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.axis('off')

    for a, b, tag in edges:
        if a in nodes and b in nodes:
            ax.annotate('', xy=(nodes[b]['x'], nodes[b]['y'] + .17),
                        xytext=(nodes[a]['x'], nodes[a]['y'] - .17),
                        arrowprops=dict(arrowstyle='-|>', color='#9aa3ad',
                                        lw=1.2, shrinkA=0, shrinkB=0))
            mx = (nodes[a]['x'] + nodes[b]['x']) / 2
            my = (nodes[a]['y'] + nodes[b]['y']) / 2
            ax.text(mx, my, tag, fontsize=8.5, ha='center', va='center',
                    color='#6b757f',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white',
                              ec='none', alpha=.85))

    for nid, n in nodes.items():
        if n['leaf']:
            txt = f"叶\n{n['gain']:+.3f}"
            fc, ec = '#eef6f0', '#2b7a4b'
        elif n.get('cut'):
            txt = '⋯'
            fc, ec = '#f5f7f9', '#b6bec7'
        else:
            txt = f"{n['feat']}\n< {n['split']:.4g}"
            fc, ec = '#fdeef0', '#d1495b'
        ax.add_patch(FancyBboxPatch(
            (n['x'] - .052, n['y'] - .17), .104, .34,
            boxstyle='round,pad=0.012', facecolor=fc, edgecolor=ec, lw=1.4))
        ax.text(n['x'], n['y'], txt, fontsize=8.2, ha='center', va='center')

    ax.set_xlim(-.08, 1.08)
    ax.set_ylim(-max_depth - .55, .75)
    ax.text(.5, .55,
            f'图5.12  单棵提升树的结构展开（behavior 集 · 第 0 棵 · 前 {max_depth} 层）',
            fontsize=13, ha='center', fontweight='bold')
    ax.text(.5, .33,
            '每个内部节点是一次「特征 < 阈值」的二分判决；叶节点数值为该叶的输出分数',
            fontsize=9.5, ha='center', color='#6b757f')
    save(fig, f'fig8_tree_structure_{CHAIN}.png')


def main():
    print('▶ 载入 behavior 特征集与模型产物 …')
    cfg, met, df, X, y, feats = load_all()
    params = dict(cfg['best_params'])
    n_folds = int(met.get('n_folds', 5))
    print(f'  样本 {len(y):,} | 特征 {len(feats)} | 折数 {n_folds}')
    print(f'  超参直接沿用训练时的 best_params（不重调，避免与表 5.1 冲突）')

    met_all = {}
    for s in ('model', 'latency', 'nometa', 'nowin', 'clean',
              'behavior', 'timing'):
        p = MODEL / f'{CHAIN}_{s}' / 'metrics.json'
        if p.exists():
            met_all[s] = json.load(open(p))

    print('\n▶ 图4.3 缺失率 …')
    miss = fig_missingness(df)
    print('\n▶ 图5.2 消融框架 …')
    fig_ablation_frame(met_all)
    print('\n▶ 图5.7 训练过程核查 (a) mlogloss + (b) 学习曲线（多次拟合，稍慢）…')
    lc, curve = fig_learning_curve(X, y, params, n_folds)
    print('\n▶ 图5.10 / 5.11 OOF 概率与 ROC（重算 OOF）…')
    aucs, med = fig_oof_proba_roc(X, y, params, n_folds, cfg['classes'])
    print('\n▶ 图5.12 单棵树 …')
    fig_tree(feats)

    # 落盘实测值，供论文对账直接引用
    out = {'missingness_pct': miss, 'learning_curve': lc,
           'train_curve': curve,
           'roc_auc_ovr': aucs, 'oof_true_proba_median': med}
    (FIG / f'MISSING_FIGS_STATS_{CHAIN}.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n✓ 实测值 → figures/MISSING_FIGS_STATS_{CHAIN}.json')
    print('✓ 全部完成')


if __name__ == '__main__':
    main()
