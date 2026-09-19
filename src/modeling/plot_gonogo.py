#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：plot_gonogo.py
@Description:
    go/no-go 建模与判定 —— 三分类 + 二分类对照 + 6 张图 + 判定书，附 validate_* 校验。

    ============================================================
    一、 这个实验要回答什么
    ============================================================
    核心假设 H1（响应延迟的三个频带）:
        传统 bot  : 亚秒 ~ 2 秒     AI agent : 2 ~ 10 秒     人类 : >10 秒

    本模块是整个课题的 **go/no-go 检验点**，应在第 1–2 周完成:
        三组分布可分 → 🟢 GO，全速推进课题一
        信号弱       → 🟡 CAUTION，扩充真值后复测
        完全重叠     → 🔴 NO-GO，转向「链上 agent 可观测性边界」负结果论文，或改课题二

    ============================================================
    二、 产出与判定依据
    ============================================================

    ┌─ 产出① — 6 张图 ────────────────────────────────────────────────────────┐
    │  fig1_latency_distribution  ★核心证据：Δt 分布 + 三频带阴影              │
    │  fig2_fast_response_ecdf     最快响应能力 dt_p05 的地址级 ECDF           │
    │  fig3_band_shares            三频带占比堆叠条                            │
    │  fig4_burstiness_entropy     节律 × 突发性散点                           │
    │  fig5_confusion              三分类混淆矩阵（RandomForest, K 折 CV）     │
    │  fig6_importance             特征重要性（蓝色 = 延迟族）                 │
    └───────────────────────────────────────────────────────────────────────────┘

    ┌─ 产出② — 判定书 GONOGO_VERDICT_{tag}.md ───────────────────────────────┐
    │  含: 判定结论 / 样本量 / 分类性能 / ★二分类对照 / KS 检验 /              │
    │      Top8 特征 / 图表索引 / 已知局限                                     │
    └───────────────────────────────────────────────────────────────────────────┘

    判定依据（三条线同时看）:
      1) 三分类 macro-F1        —— 阈值 config.GONOGO_F1_GO / _CAUTION
      2) agent 类召回率          —— 阈值 config.GONOGO_AGENT_RECALL_GO / _MIN
      3) ★ 二分类对照实验的漏检率 —— 对标 Web 域 arXiv 2607.26935 的 34.5% / 39.1%

    ★ 二分类对照实验是**论文核心叙事**:
      只用 human + bot 训一个二分类器（模拟已有工作），再拿它去分 agent。
      无论结果高于还是低于 Web 域，都是可写进论文的结论:
        高 → 链上问题更严重；低 → 链上 agent 更接近 bot，需解释为什么。

    ============================================================
    三、 关键的绘图/统计处理
    ============================================================
      1. dt_p05 可以**合法地等于 0**（同秒多笔 —— bot 的典型形态）:
         · 不能丢弃（会把 bot 组的主体信号删掉）
         · 不能直接取 log
         ⇒ 做法: clip 到 0.1s 下限后取 log，保留底部质量
      2. Δt==0 的占比单独统计并标在图例上 —— 它本身就是强 bot 信号
      3. 每地址下采样 max_per_addr=500，避免超活跃地址主导分布
      4. 中文字体缺 U+2212 字形 ⇒ 轴标签用 ASCII 连字符、刻度用明文（1s/10s/1min）

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🏁 **管线终点** —— 上承 src/feature/compute_features.py 的特征表，
        产出 go/no-go 判定书。第一周只需跑通到这里，就能决定课题做不做得下去。
'''

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils.common import log, setup_matplotlib_cjk  # noqa: E402
# 🔴 2026-09-04 修：原来从 compute_features 导入（旧特征名，无 f_ 前缀、24 个），
#    而 f_ensemble 产出的宽表用 f_ 前缀、57 个 —— 两者零匹配，
#    导致 --chain base 时筛出 0 个特征，SimpleImputer 直接抛
#    "Found array with 0 feature(s)"。统一改为从 f_ensemble 导入。
from src.feature.f_ensemble import (BEHAVIOR_FEATURES,  # noqa: E402
                                    LATENCY_ONLY_FEATURES,
                                    MODEL_FEATURES)

setup_matplotlib_cjk()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


# ================================================================
#  GoNoGoAnalyzer — 三分类建模、出图与 go/no-go 判定
# ================================================================
class GoNoGoAnalyzer(object):
    """
    go/no-go 分析器

    核心函数 (共 4 个):
      ① run_cv(feats, name)        三分类交叉验证（LogReg + RandomForest）
      ② run_binary_control(feats)  ★二分类对照实验（论文核心叙事）
      ③ make_figures()             出 6 张图
      ④ run()                      ①②③ + 判定书

    绘图函数: fig1_latency_distribution / fig2_fast_response_ecdf /
             fig3_band_shares / fig4_burstiness_entropy /
             fig5_confusion / fig6_importance

    统计函数: pairwise_tests（两两 KS + Wasserstein）

    校验函数: validate_features

    Args:
        tag: 数据标签（链名或 'synthetic'）
    """

    GROUP_ORDER = ['bot', 'agent', 'human']

    # dt_p05 可以合法为 0，clip 到这个下限后再取 log（见模块文档 三.1）
    LOG_FLOOR = 0.1
    # 每地址下采样上限，避免超活跃地址主导分布
    MAX_DT_PER_ADDR = 500
    # Web 域对照基准（arXiv 2607.26935）
    WEB_DOMAIN_MLP = 39.1
    WEB_DOMAIN_SAINT = 34.5
    # KS 显著性判定线
    KS_SIGNIFICANT = 0.3

    def __init__(self, tag: str = None):
        """
        初始化

        Args:
            tag: 数据标签；缺省 config.PRIMARY_CHAIN
        Returns:
            None
        """
        self.tag = tag or config.PRIMARY_CHAIN
        self.colors = {g: config.GROUPS[g][2] for g in config.GROUPS}
        self.names = {g: config.GROUPS[g][1] for g in config.GROUPS}
        self.df = pd.DataFrame()
        self.pooled = {}
        self.zero_frac = {}
        self.all_res = {}
        self.lat_res = {}
        self.beh_res = {}
        self.binctl = {}
        self.tests = pd.DataFrame()

    # ============================================================
    #  通用工具 / 数据载入
    # ============================================================
    def col(self, name: str) -> str:
        """
        列名解析 —— 兼容新旧两套特征命名

        【为什么需要】 f_ensemble 产出的列名带 f_ 前缀（f_dt_p05），
                     而本模块的绘图代码写于重构前，用的是裸名（dt_p05）。
                     旧的 features_synthetic.csv 仍是裸名，两者都要能跑。
        【规则】 先找带 f_ 前缀的，没有再找裸名；都没有则抛 KeyError 并
                提示是哪一个特征缺了 —— 比 pandas 原生的 KeyError 好定位。

        Args:
            name: 裸特征名，如 'dt_p05'
        Returns:
            str: 本 DataFrame 中实际存在的列名
        """
        for cand in (f'f_{name}', name):
            if cand in self.df.columns:
                return cand
        raise KeyError(f'特征表里既没有 f_{name} 也没有 {name} —— '
                       f'先跑 python3 src/feature/f_ensemble.py')

    @staticmethod
    def _safe_float(v, default: float = float('nan')):
        """安全转 float。"""
        try:
            fv = float(v)
            return default if fv != fv else fv
        except (TypeError, ValueError):
            return default

    def _log_clip(self, arr) -> np.ndarray:
        """clip 到 LOG_FLOOR 后取 log10（保留 dt_p05=0 的底部质量）。"""
        return np.log10(np.clip(np.asarray(arr, dtype=float), self.LOG_FLOOR, None))

    def load_features(self) -> pd.DataFrame:
        """载入特征表（data/features/features_{tag}.csv）。"""
        p = config.DATA_FEAT / f'features_{self.tag}.csv'
        if not p.exists():
            log(f'找不到 {p}，先跑 src/feature/compute_features.py', 'err')
            sys.exit(1)
        df = pd.read_csv(p)
        self.df = df[df['group'].isin(self.GROUP_ORDER)].copy()
        return self.df

    def load_pooled_dt(self) -> tuple:
        """
        从原始 JSON 重新汇集所有 Δt（每地址下采样，避免大地址主导分布）

        Returns:
            (pooled_positive, zero_frac)
              pooled_positive —— Δt > 0 的部分，供对数刻度绘图
              zero_frac       —— Δt == 0（同秒/同区块）的占比，单独报告，
                                 因为它本身就是强 bot 信号，不能默默丢掉
        """
        base = config.DATA_RAW / self.tag
        rng = np.random.default_rng(config.RANDOM_SEED)
        pooled, zero_frac = {}, {}

        for group in self.GROUP_ORDER:
            gdir = base / group
            if not gdir.exists():
                continue
            vals, n_zero, n_all = [], 0, 0
            for fp in gdir.glob('*.json'):
                try:
                    txs = json.loads(fp.read_text())
                except json.JSONDecodeError:
                    continue
                if len(txs) < config.MIN_TX_FOR_FEATURES:
                    continue
                ts = np.sort(np.array([t['timeStamp'] for t in txs], dtype=float))
                dt = np.diff(ts)
                n_all += len(dt)
                n_zero += int((dt == 0).sum())
                dt = dt[dt > 0]
                if len(dt) == 0:
                    continue
                if len(dt) > self.MAX_DT_PER_ADDR:      # 3. 下采样
                    dt = rng.choice(dt, self.MAX_DT_PER_ADDR, replace=False)
                vals.append(dt)
            if vals:
                pooled[group] = np.concatenate(vals)
                zero_frac[group] = n_zero / n_all if n_all else 0.0

        self.pooled, self.zero_frac = pooled, zero_frac
        return pooled, zero_frac

    # ============================================================
    #  绘图
    # ============================================================
    def fig1_latency_distribution(self) -> None:
        """★ 核心图：Δt 分布 + 三频带阴影。三条曲线分离 = H1 成立 = GO。"""
        fig, ax = plt.subplots(figsize=(10, 5.5))
        bands = [
            (np.log10(0.5), np.log10(config.BAND_BOT[1]), '#b03636', 'bot 带 <2s'),
            (np.log10(config.BAND_AGENT[0]), np.log10(config.BAND_AGENT[1]),
             '#2f6fb0', 'agent 带 2–10s'),
            (np.log10(config.BAND_HUMAN[0]), 6, '#2b7a4b', 'human 带 >10s'),
        ]
        for lo, hi, c, _ in bands:
            ax.axvspan(lo, hi, color=c, alpha=0.06, zorder=0)
        for lo, _, _, _ in bands[1:]:
            ax.axvline(lo, color='#767d87', ls='--', lw=1, zorder=1)

        bins = np.linspace(-0.5, 6, 130)
        for g in self.GROUP_ORDER:
            if g not in self.pooled:
                continue
            x = np.log10(self.pooled[g])
            zf = self.zero_frac.get(g, 0.0)
            lbl = f'{self.names[g]}  (n={len(x):,}'
            lbl += f', 同秒 {zf:.0%})' if zf >= 0.005 else ')'
            ax.hist(x, bins=bins, density=True, histtype='step', lw=2.0,
                    color=self.colors[g], label=lbl, zorder=3)
            ax.hist(x, bins=bins, density=True, color=self.colors[g],
                    alpha=0.12, zorder=2)

        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(['1s', '10s', '100s', '17min', '2.8h', '28h'])
        ax.set_xlabel('相邻交易间隔 Δt（对数刻度）')
        ax.set_ylabel('概率密度')
        ax.set_title('图5.3  三类账户相邻活动间隔 Δt 的分布与三频带\n'
                     '（合并分布；正文引用的频带占比为组内均值口径，见 5.4 节）', fontsize=12, pad=14)

        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor=c, alpha=0.25, label=lb) for _, _, c, lb in bands]
        labels += [lb for _, _, _, lb in bands]
        ax.legend(handles, labels, fontsize=9, loc='upper right')
        fig.savefig(config.FIG_DIR / f'fig1_latency_distribution_{self.tag}.png')
        plt.close(fig)

    def fig2_fast_response_ecdf(self) -> None:
        """图2：最快响应能力 dt_p05 的地址级 ECDF（0 值 clip 到 0.1s）。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        for g in self.GROUP_ORDER:
            s = self.df.loc[self.df['group'] == g, self.col('dt_p05')].dropna()
            if s.empty:
                continue
            x = np.sort(np.clip(s.values, self.LOG_FLOOR, None))
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where='post', lw=2, color=self.colors[g],
                    label=f'{self.names[g]}  (n={len(x)})')
        ax.set_xscale('log')
        ax.axvline(2, color='#767d87', ls='--', lw=1)
        ax.axvline(10, color='#767d87', ls='--', lw=1)
        # 4. 明文刻度：避免 mathtext 上标负号（中文字体缺 U+2212）
        ax.set_xticks([0.1, 1, 2, 10, 60, 600, 3600, 86400])
        ax.set_xticklabels(['0.1s', '1s', '2s', '10s', '1min', '10min', '1h', '1d'])
        ax.minorticks_off()
        ax.set_xlabel('dt_p05：该地址最快响应能力（秒，第5百分位间隔；0 值截断到 0.1s）')
        ax.set_ylabel('累积占比')
        ax.set_title('图5.4  地址级最快响应能力 Δt(p05) 的经验累积分布', pad=12)
        ax.legend(fontsize=9)
        fig.savefig(config.FIG_DIR / f'fig2_fast_response_ecdf_{self.tag}.png')
        plt.close(fig)

    def fig3_band_shares(self) -> None:
        """图3：三频带占比堆叠条。agent 组在 2–10s 带显著更高 → 支持 H1。"""
        cols = [self.col(c) for c in ('frac_lt2s', 'frac_2to10s', 'frac_gt10s')]
        m = (self.df.groupby('group')[cols].mean()
             .reindex(self.GROUP_ORDER).dropna(how='all'))
        if m.empty:
            return

        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        bottom = np.zeros(len(m))
        for col, c, lb in zip(cols, ['#b03636', '#2f6fb0', '#2b7a4b'],
                              config.BAND_LABELS):
            v = m[col].values
            ax.barh(range(len(m)), v, left=bottom, color=c, alpha=0.85, label=lb)
            for i, (val, b) in enumerate(zip(v, bottom)):
                if val > 0.06:
                    ax.text(b + val / 2, i, f'{val:.0%}', ha='center', va='center',
                            color='white', fontsize=10, fontweight='bold')
            bottom += v

        ax.set_yticks(range(len(m)))
        ax.set_yticklabels([self.names[g] for g in m.index])
        ax.set_xlim(0, 1)
        ax.set_xlabel('交易间隔落在各频带的平均占比')
        ax.set_title('图5.5  三类账户的三频带活动占比对比（组内均值）\n'
                     '若 agent 组在 2–10s 带显著更高 → 支持 H1', pad=12)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(axis='y', visible=False)
        fig.savefig(config.FIG_DIR / f'fig3_band_shares_{self.tag}.png')
        plt.close(fig)

    def fig4_burstiness_entropy(self) -> None:
        """图4：节律 × 突发性散点。"""
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for g in self.GROUP_ORDER:
            s = self.df[self.df['group'] == g]
            ax.scatter(s[self.col('entropy_24h')], s[self.col('burstiness')], s=26, alpha=0.55,
                       color=self.colors[g], edgecolors='none', label=self.names[g])
        ax.set_xlabel('24 小时活动熵（1 = 全天均匀，低 = 有昼夜节律）')
        # ASCII 连字符，避免中文字体缺 U+2212 字形
        ax.set_ylabel('Burstiness  B = (sd-mean)/(sd+mean)')
        ax.axhline(0, color='#c9ced6', lw=1)
        ax.set_title('图5.6  突发性系数与 24 小时活动熵的联合分布\n'
                     '预期：bot 右下（均匀+周期），human 左上，agent 右上（均匀+突发）',
                     pad=12)
        ax.legend(fontsize=9)
        fig.savefig(config.FIG_DIR / f'fig4_burstiness_entropy_{self.tag}.png')
        plt.close(fig)

    def fig5_confusion(self) -> None:
        """图5：三分类混淆矩阵。重点看 bot↔agent 那一格。"""
        res = self.all_res
        if '_cm' not in res:
            return
        cm = res['_cm'].astype(float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
        labels = res['_labels']

        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([self.names[g] for g in labels], rotation=18,
                           ha='right', fontsize=9)
        ax.set_yticklabels([self.names[g] for g in labels], fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cmn[i, j]:.2f}\n({int(cm[i, j])})',
                        ha='center', va='center',
                        color='white' if cmn[i, j] > 0.5 else '#1a1d21', fontsize=10)
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')
        ax.set_title(f"（附·非论文用图）三分类混淆矩阵（RandomForest, {res['n_splits']}折CV）\n"
                     f"macro-F1 = "
                     f"{self._safe_float(res.get('RandomForest_macro_f1')):.3f}", pad=12)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045)
        fig.savefig(config.FIG_DIR / f'fig5_confusion_{self.tag}.png')
        plt.close(fig)

    def fig6_importance(self, top: int = 18) -> None:
        """图6：特征重要性。蓝色 = 延迟族，越靠上说明 H1 越成立。"""
        imp = self.all_res.get('_importance')
        if not imp:
            return
        s = pd.Series(imp).sort_values(ascending=True).tail(top)
        fig, ax = plt.subplots(figsize=(8, max(4.5, 0.32 * len(s))))
        is_lat = [f in LATENCY_ONLY_FEATURES for f in s.index]
        ax.barh(range(len(s)), s.values,
                color=['#2f6fb0' if b else '#c9ced6' for b in is_lat])
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=9)
        ax.set_xlabel('RandomForest 特征重要性')
        ax.set_title('（附·非论文用图）RandomForest 特征重要性（蓝色 = 延迟族）\n'
                     '蓝色越靠上，H1「响应延迟是核心信号」越成立', pad=12)
        ax.grid(axis='y', visible=False)
        fig.savefig(config.FIG_DIR / f'fig6_importance_{self.tag}.png')
        plt.close(fig)

    def make_figures(self) -> None:
        """③ 出全部 6 张图。"""
        log('出图 1-4 ...', 'info')
        self.load_pooled_dt()
        if self.pooled:
            self.fig1_latency_distribution()
        self.fig2_fast_response_ecdf()
        self.fig3_band_shares()
        self.fig4_burstiness_entropy()
        if self.all_res:
            self.fig5_confusion()
            self.fig6_importance()

    # ============================================================
    #  统计检验
    # ============================================================
    def pairwise_tests(self) -> pd.DataFrame:
        """
        对 log10(dt_p05) 做两两 KS 检验 + Wasserstein 距离

        ⚠️ dt_p05 可以合法地等于 0（同秒多笔 —— bot 的典型形态）:
           不能丢弃（会把 bot 组的主体信号删掉），也不能直接取 log。
           做法: clip 到 0.1s 下限后取 log，保留底部质量。

        Returns:
            pd.DataFrame: [对比, n_a, n_b, KS统计量, p值, Wasserstein距离, 中位数比]
        """
        rows = []
        for i, a in enumerate(self.GROUP_ORDER):
            for b in self.GROUP_ORDER[i + 1:]:
                xa = self.df.loc[self.df['group'] == a, self.col('dt_p05')].dropna()
                xb = self.df.loc[self.df['group'] == b, self.col('dt_p05')].dropna()
                if len(xa) < 5 or len(xb) < 5:
                    continue
                la, lb = self._log_clip(xa.values), self._log_clip(xb.values)
                ks = stats.ks_2samp(la, lb)
                rows.append({
                    '对比': f'{a} vs {b}',
                    'n_a': len(la), 'n_b': len(lb),
                    'KS统计量': round(float(ks.statistic), 3),
                    'p值': f'{ks.pvalue:.2e}',
                    'Wasserstein距离': round(
                        float(stats.wasserstein_distance(la, lb)), 3),
                    '中位数比': round(float(10 ** (np.median(lb) - np.median(la))), 2),
                })
        self.tests = pd.DataFrame(rows)
        return self.tests

    # ============================================================
    #  ① 三分类交叉验证
    # ============================================================
    def run_cv(self, feats: list, name: str) -> dict:
        """
        ① 三分类交叉验证（LogReg + RandomForest）

        【数据源】 data/features/features_{tag}.csv

        【指标含义】 一句话: "三类到底分不分得开"
                    macro-F1 是主指标（类不平衡，micro 会被大类主导）。

        【返回字段】 feature_set / n_features / n_splits /
                    {LogReg,RandomForest}_macro_f1 / _recall_{group} /
                    _cm / _labels / _importance

        【阈值解读】 见 config.GONOGO_F1_GO(0.70) / _CAUTION(0.55)

        Args:
            feats: 参与建模的特征列
            name:  特征集名（用于日志与判定书）
        Returns:
            dict
        """
        use = [f for f in feats if f in self.df.columns]
        X = self.df[use].replace([np.inf, -np.inf], np.nan).values
        y = self.df['group'].values

        n_min = pd.Series(y).value_counts().min()
        if n_min < 5:
            log(f'[{name}] 最小类只有 {n_min} 个样本，跳过交叉验证', 'warn')
            return {}
        n_splits = int(min(5, n_min))

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=config.RANDOM_SEED)
        out = {'feature_set': name, 'n_features': len(use), 'n_splits': n_splits}

        models = {
            'LogReg': make_pipeline(
                SimpleImputer(strategy='median'), StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight='balanced',
                                   random_state=config.RANDOM_SEED)),
            'RandomForest': make_pipeline(
                SimpleImputer(strategy='median'),
                RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                       class_weight='balanced_subsample',
                                       random_state=config.RANDOM_SEED, n_jobs=-1)),
        }

        for mname, model in models.items():
            pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            out[f'{mname}_macro_f1'] = float(f1_score(y, pred, average='macro'))
            for g in self.GROUP_ORDER:
                if g in set(y):
                    out[f'{mname}_recall_{g}'] = float(
                        recall_score(y, pred, labels=[g], average='macro',
                                     zero_division=0))
            if mname == 'RandomForest':
                out['_cm'] = confusion_matrix(y, pred, labels=self.GROUP_ORDER)
                out['_labels'] = self.GROUP_ORDER
                rf = models['RandomForest'].fit(X, y)
                out['_importance'] = dict(zip(
                    use, rf.named_steps['randomforestclassifier'].feature_importances_))
        return out

    # ============================================================
    #  ② 二分类对照实验（论文核心叙事）
    # ============================================================
    def run_binary_control(self, feats: list) -> dict:
        """
        ② ★ 二分类对照实验 —— 论文核心叙事

        【数据源】 同 ①，但只用 human + bot 训练

        【指标含义】 一句话: "已有的二分类方法会把 agent 判成什么"
                    只用 human + bot 训一个二分类器（模拟已有工作），
                    再拿它去分 agent，统计 agent 被分到哪一类。

        【返回字段】 n_agent / as_human_pct / as_bot_pct /
                    web_domain_mlp / web_domain_saint

        【阈值解读】 对标 Web 域 arXiv 2607.26935:
                    MLP 二分类漏检 39.1% / SAINT 漏检 34.5%

        【如何使用】 🎯 无论本实验数字高于还是低于 Web 域，都是可写进论文的结论:
                   · 高 → 链上问题更严重，二分类方法失效更彻底
                   · 低 → 链上 agent 行为更接近 bot，需解释为什么
                   这个「两头都不亏」的设计，正是 §2.3 立论方式的实现。

        Args:
            feats: 参与建模的特征列
        Returns:
            dict；样本不足时返回 {}
        """
        use = [f for f in feats if f in self.df.columns]
        tr = self.df[self.df['group'].isin(['human', 'bot'])]
        ag = self.df[self.df['group'] == 'agent']
        if len(ag) < 5 or tr['group'].nunique() < 2:
            return {}

        model = make_pipeline(
            SimpleImputer(strategy='median'),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                   class_weight='balanced_subsample',
                                   random_state=config.RANDOM_SEED, n_jobs=-1))
        model.fit(tr[use].replace([np.inf, -np.inf], np.nan).values,
                  tr['group'].values)
        pred = model.predict(ag[use].replace([np.inf, -np.inf], np.nan).values)

        n = len(pred)
        return {
            'n_agent': n,
            'as_human_pct': float((pred == 'human').sum() / n * 100),
            'as_bot_pct': float((pred == 'bot').sum() / n * 100),
            'web_domain_mlp': self.WEB_DOMAIN_MLP,
            'web_domain_saint': self.WEB_DOMAIN_SAINT,
        }

    # ============================================================
    #  判定书
    # ============================================================
    def verdict(self, synthetic: bool = False) -> str:
        """
        生成 go/no-go 判定书（Markdown）

        Args:
            synthetic: 是否为合成数据（会加醒目免责声明）
        Returns:
            str: Markdown 文本
        """
        f1_all = self._safe_float(self.all_res.get('RandomForest_macro_f1'))
        f1_lat = self._safe_float(self.lat_res.get('RandomForest_macro_f1'))
        f1_beh = self._safe_float(self.beh_res.get('RandomForest_macro_f1'))
        rec_beh = self._safe_float(self.beh_res.get('RandomForest_recall_agent'))

        # 🔴 判定以最严的 behavior 集为准（与 train_xgb 一致）。
        #    全特征集含元特征/窗口特征/缺失模式三条捷径，用它判定会高估。
        #    behavior 跑不出来时（样本太少）才退回全特征集。
        if not np.isnan(f1_beh) and not np.isnan(rec_beh):
            f1_j, rec_agent, judged_on = f1_beh, rec_beh, '纯行为集'
        else:
            f1_j = f1_all
            rec_agent = self._safe_float(
                self.all_res.get('RandomForest_recall_agent'))
            judged_on = '全部特征（behavior 集不可用，判定偏乐观）'

        if np.isnan(f1_j) or np.isnan(rec_agent):
            flag, txt = '⚠️ 无法判定', '样本不足，无法完成交叉验证。每组至少各需 30+ 个地址。'
        elif (f1_j >= config.GONOGO_F1_GO
              and rec_agent >= config.GONOGO_AGENT_RECALL_GO):
            flag, txt = '🟢 GO', '三分类信号成立，H1 得到支持。全速推进课题一。'
        elif (f1_j >= config.GONOGO_F1_CAUTION
              and rec_agent >= config.GONOGO_AGENT_RECALL_MIN):
            flag, txt = '🟡 CAUTION', (
                '信号存在但不够强。先扩充真值（尤其 Olas 高质量正样本）、'
                '补图特征与序列表示，2 周后复测。暂不换题。')
        else:
            flag, txt = '🔴 NO-GO', (
                '延迟与基础行为特征无法区分三类。两条路：'
                '(a) 转向「链上 agent 可观测性边界」的负结果论文——这仍是有价值的贡献；'
                '(b) 改做课题二 RWA。')

        L = [f'# go/no-go 判定：{flag}\n']
        if synthetic:
            L.append('> ## ⚠️⚠️ 这是合成数据的结果，**不构成任何真实判定** ⚠️⚠️\n'
                     '> 合成数据是按 H1 假设生成的，跑出 GO 是必然的，'
                     '只能用来验证管线通不通。\n'
                     '> 真实判定跑 `./bin/run_all.sh`（完整，约 2.5 小时）。\n')
        L.append(f'**结论**：{txt}\n\n---\n')

        L.append('## 1. 样本量\n')
        L.append(self.df['group'].value_counts().rename('地址数')
                 .to_frame().to_markdown() + '\n')

        L.append('\n## 2. 分类性能\n')
        L.append('| 特征集 | 特征数 | LogReg macro-F1 | RF macro-F1 | RF agent 召回 |')
        L.append('|---|---|---|---|---|')
        for r, nm in ((self.lat_res, '仅延迟族'),
                      (self.beh_res, '★纯行为集（判定依据）'),
                      (self.all_res, '全部特征（含捷径，偏乐观）')):
            if r:
                L.append(f"| {nm} | {r['n_features']} | "
                         f"{self._safe_float(r.get('LogReg_macro_f1')):.3f} | "
                         f"{self._safe_float(r.get('RandomForest_macro_f1')):.3f} | "
                         f"{self._safe_float(r.get('RandomForest_recall_agent')):.3f} |")
        L.append(f'\n判定线：macro-F1 ≥ {config.GONOGO_F1_GO} 且 agent 召回 ≥ '
                 f'{config.GONOGO_AGENT_RECALL_GO} → GO\n')
        L.append(f'\n**本次判定依据：{judged_on}** —— '
                 f'全特征集含元特征/窗口特征/缺失模式三条捷径，'
                 f'用它判定会高估信号强度。\n')
        if not np.isnan(f1_lat) and not np.isnan(f1_all):
            L.append(f"\n**延迟族单独贡献**：仅用 {self.lat_res['n_features']} 个"
                     f'延迟特征就达到 macro-F1 {f1_lat:.3f}，全部特征 {f1_all:.3f}。'
                     f'差值 {f1_all - f1_lat:+.3f} —— 差值越小，说明延迟越是核心信号。\n')

        if self.binctl:
            b = self.binctl
            L.append('\n## 3. ★ 二分类对照实验（论文核心叙事）\n')
            L.append('用 human/bot 二分类器去分 agent 地址：\n')
            L.append('| 结果 | 占比 |')
            L.append('|---|---|')
            L.append(f"| agent → 被判为 **human** | **{b['as_human_pct']:.1f}%** |")
            L.append(f"| agent → 被判为 **bot** | {b['as_bot_pct']:.1f}% |")
            L.append(f"\n对标 Web 域 (arXiv 2607.26935)：MLP 二分类漏检 "
                     f"{b['web_domain_mlp']}%、SAINT 漏检 {b['web_domain_saint']}%。\n")
            L.append('> 无论本实验数字高于还是低于 Web 域，都是可写进论文的结论：\n'
                     '> 高 → 链上问题更严重；低 → 链上 agent 更接近 bot，需解释为什么。\n')

        if not self.tests.empty:
            L.append('\n## 4. 分布差异检验（基于 log10(dt_p05)，0 值截断到 0.1s）\n')
            L.append(self.tests.to_markdown(index=False) + '\n')
            L.append(f'\nKS 统计量 > {self.KS_SIGNIFICANT} 且 p < 0.01 '
                     f'视为分布显著可分。\n')

        if '_importance' in self.all_res:
            top = (pd.Series(self.all_res['_importance'])
                   .sort_values(ascending=False).head(8))
            L.append('\n## 5. Top 8 特征\n')
            L.append('| 特征 | 重要性 | 属于延迟族 |')
            L.append('|---|---|---|')
            for f, v in top.items():
                L.append(f"| `{f}` | {v:.4f} | "
                         f"{'✅' if f in LATENCY_ONLY_FEATURES else ''} |")

        L.append('\n## 6. 图表\n')
        for i, d in enumerate([
            'fig1_latency_distribution ★ 核心图：Δt 分布 + 三频带',
            'fig2_fast_response_ecdf   最快响应能力的地址级分布',
            'fig3_band_shares          三频带占比',
            'fig4_burstiness_entropy   节律 × 突发性',
            'fig5_confusion            三分类混淆矩阵',
            'fig6_importance           特征重要性',
        ], 1):
            L.append(f'{i}. `figures/{d}`')

        L.append('\n## 7. 本实验的已知局限（写论文时必须交代）\n')
        L.append('- **Δt 是代理量，不是真正的 event-response latency。** '
                 '链上无法历史回溯触发事件（HTTP 请求在链下、mempool 难回放）。'
                 '完整版需引入 mempool 数据或合约事件配对。')
        L.append('- **人类样本有噪声。** CEX 提现地址只是弱先验，需做敏感性分析。')
        L.append('- **Olas 正样本量小但质量最高**，x402/ERC-8004 正样本量大但含大量空壳'
                 '（实证：ERC-8004 仅 3–15% 有有效注册）。两者不可混用而不分层。')
        L.append('- **出块时间是分辨率下限。** Base 2s 出块意味着 <2s 的差异不可分辨；'
                 '以太坊 12s 出块则完全无法区分 bot 与 agent 频带。')
        return '\n'.join(L)

    # ============================================================
    #  ④ 主流程
    # ============================================================
    def run(self, synthetic: bool = False) -> str:
        """
        ④ 完整流程：载入 → 建模 → 出图 → 判定书

        Args:
            synthetic: 是否为合成数据
        Returns:
            str: 判定书 Markdown 文本
        """
        log(f'分析：{self.tag}', 'step')
        self.load_features()
        log(f'载入 {len(self.df)} 个地址', 'info')
        self.validate_features()

        log('交叉验证 ...', 'info')
        self.lat_res = self.run_cv(LATENCY_ONLY_FEATURES, '仅延迟族')
        self.all_res = self.run_cv(MODEL_FEATURES, '全部特征')
        # 🔴 与 train_xgb 口径对齐：go/no-go 应以最严的 behavior 集为准，
        #    它堵掉了协议形态/采集窗口/缺失模式三条捷径（见 f_ensemble 文档）
        self.beh_res = self.run_cv(BEHAVIOR_FEATURES, '纯行为集（主结论）')
        self.binctl = self.run_binary_control(MODEL_FEATURES)

        self.make_figures()
        self.pairwise_tests()

        md = self.verdict(synthetic)
        out = config.FIG_DIR / f'GONOGO_VERDICT_{self.tag}.md'
        out.write_text(md, encoding='utf-8')

        print('\n' + '=' * 66)
        print(md.split('\n---\n')[0])
        print('=' * 66)
        log(f'完整判定书：{out}', 'ok')
        log(f'图表目录：{config.FIG_DIR}', 'ok')
        return md

    # ============================================================
    #  数据校验
    # ============================================================
    @staticmethod
    def _emit(context: str, issues: list, warns: list) -> bool:
        """统一输出校验结论。Returns: 是否通过（无 issue）。"""
        if issues:
            log(f'[{context}] ❌ 未通过（{len(issues)} 项问题）', 'err')
            for i in issues:
                print(f'    · {i}')
        elif warns:
            log(f'[{context}] ⚠️ 通过但有 {len(warns)} 项提示', 'warn')
        else:
            log(f'[{context}] ✅ 通过', 'ok')
        for w in warns:
            print(f'    · {w}')
        return not issues

    def validate_features(self, context: str = '建模输入') -> bool:
        """
        校验建模输入

        检查项:
          1. 非空 / 建模特征列齐全
          2. ⚠️ 三组是否齐全（三分类需要三组）
          3. ⚠️ 最小类样本量是否够做 K 折 CV
          4. ⚠️ 类别不平衡
          5. ⚠️ 全 NaN 的特征列
          6. ⚠️ 缺 agent 组则二分类对照实验做不了

        Args:
            context: 日志上下文名
        Returns:
            bool
        """
        issues, warns = [], []
        df = self.df

        if df is None or df.empty:
            issues.append('特征表为空')
            return self._emit(context, issues, warns)

        miss = [c for c in MODEL_FEATURES if c not in df.columns]
        if miss:
            issues.append(f'缺建模特征列: {miss}')
            return self._emit(context, issues, warns)

        vc = df['group'].value_counts()
        lost = set(self.GROUP_ORDER) - set(vc.index)
        if lost:
            warns.append(f'缺少分组 {sorted(lost)} —— 三分类需要三组齐全')
        if 'agent' not in vc.index:
            warns.append('缺 agent 组 —— ★二分类对照实验无法进行')

        if len(vc):
            n_min = int(vc.min())
            if n_min < 5:
                issues.append(f'最小类仅 {n_min} 个样本，无法做交叉验证')
            elif n_min < 30:
                warns.append(f'最小类仅 {n_min} 个样本，K 折 CV 结果会不稳定')
            if len(vc) > 1 and vc.max() / vc.min() > 5:
                warns.append(f'类别严重不平衡 {dict(vc)} —— '
                             f'已用 class_weight，评估请看 macro-F1 而非 accuracy')

        allnan = [c for c in MODEL_FEATURES if df[c].isna().all()]
        if allnan:
            warns.append(f'{len(allnan)} 个特征全为 NaN: {allnan[:5]} '
                         f'—— 会被 SimpleImputer 填成常数，等于无效特征')
        return self._emit(context, issues, warns)


def main():
    ap = argparse.ArgumentParser(description='go/no-go 建模与判定')
    ap.add_argument('--chain', default=config.PRIMARY_CHAIN,
                    choices=list(config.CHAINS))
    ap.add_argument('--synthetic', action='store_true', help='使用合成数据')
    args = ap.parse_args()
    tag = 'synthetic' if args.synthetic else args.chain
    GoNoGoAnalyzer(tag=tag).run(synthetic=args.synthetic)


if __name__ == '__main__':
    main()
