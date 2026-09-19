#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：compute_features.py
@Description:
    行为特征工程 —— 六大特征族统一封装，CSV 落盘，附 validate_* 数据校验。

    ============================================================
    一、 核心假设与两个延迟代理
    ============================================================
    核心假设 H1（响应延迟的三个频带）:
        传统 bot   : 亚秒 ~ 2 秒    （预编译逻辑，抢区块）
        AI agent   : 2 ~ 10 秒      （★ LLM 推理延迟 + API 往返）
        人类        : >10 秒，方差极大

    两个可观测代理:
      M1  相邻交易间隔 Δt
          最简单、无歧义。低分位数（p01/p05）刻画「这个地址最快能多快发下一笔」，
          直接对应 H1 的三个频带。这是 go/no-go 的主指标。

      M2  同目标响应间隔
          只统计对同一目标合约(to)的连续交互间隔。语义上更接近
          「观察 → 决策 → 行动」的循环，噪声更低但样本更少。

    🔴 诚实声明: 链上无法直接观测「触发事件」（HTTP 请求、mempool 观察都在链下
       或难以历史回溯），所以这是**代理量而非真正的 event-response latency**。
       论文里必须写清这一点；完整版需引入 mempool 数据或合约事件配对。

    🔴 分辨率天花板: EVM 区块时间戳是整秒。实测 BSC 多个区块共享同一秒
       （真实出块 ~0.45s），Base 每块 +2 秒。
       ⇒ 链上可观测时间分辨率 = max(出块时间, 1 秒)，**亚秒差异永远不可见**。
       且 Base 上 Δt=2s（连续区块）正好压在 agent 带下边界 ——
       实测一个纯 bot 的 frac_2to10s = 1.00，是假阳性。
       对策: same_block_ratio / consec_block_ratio 在亚区块尺度上更有判别力。

    ============================================================
    二、 六大特征族（每族一个方法，便于消融实验按族剔除）
    ============================================================

    ┌─ 族 1 — 时序节律 (f_rhythm) ─────────────────────────────────────────────┐
    │  burstiness / memory_coef / same_block_ratio / consec_block_ratio        │
    │  longest_silence_h / entropy_24h / entropy_dow / night_ratio             │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ burstiness B=(σ−μ)/(σ+μ)∈[-1,1]: -1 完全周期(定时脚本) / 0 泊松 / →1 事件驱动
      ※ entropy_24h: 人类低（有睡眠段）/ bot 高（全天均匀）/ agent 中等偏事件驱动

    ┌─ 族 2 — Gas 策略 (f_gas) ────────────────────────────────────────────────┐
    │  gas_price_cv / gas_price_mode_share                                     │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 已有论文证明**钱包软件可被 gas 策略指纹化**。
        mode_share 高 = gasPrice 恒定 = 硬编码脚本。

    ┌─ 族 3 — Nonce 并发 (f_nonce) ───────────────────────────────────────────┐
    │  nonce_gap_ratio                                                         │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 并发 pending nonce 是程序化操作的强信号（人类钱包一般一笔确认再发下一笔）。

    ┌─ 族 4 — 交互结构 (f_interaction) ───────────────────────────────────────┐
    │  unique_to_ratio / method_entropy / failed_ratio                        │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 脚本高重复、人类高多样。method_entropy 依赖 sighash/methodId 字段。

    ┌─ 族 5 — 金额偏好 (f_value) ─────────────────────────────────────────────┐
    │  value_zero_ratio / round_value_ratio                                   │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 「整数偏好」= 金额是 0.01 ETH 的整数倍，是脚本特征。

    ┌─ 族 6 — ★ 反应延迟 (f_latency) 核心创新特征 ────────────────────────────┐
    │  dt_min/p01/p05/p10/p25/p50/p75/mean/std / log_dt_median                │
    │  frac_lt2s / frac_2to10s / frac_gt10s / m2_p05 / m2_p50 / m2_n          │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 🎯 为什么最值得下注:
        ① 对应真实物理约束 —— LLM 推理是 bot 和人类都不占据的频带
        ② 难以规避 —— 可以伪造 gas 策略，但没法让 LLM 推理变快；
           加人为延迟就丧失了 agent 抢先执行的经济优势（evasion-cost 论证）
        ③ 正面回应「链上特征与 Web 域实质不同在哪」

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. 交易数 < MIN_TX_FOR_FEATURES(20) 的地址直接剔除（统计量不可靠）
      2. 按 (timeStamp, txIndex) 升序排序后算 Δt，负值剔除
      3. Δt 有效样本 < 5 ⇒ 该地址剔除
      4. 建模特征排除规模类（n_tx / span_days / tx_per_day）——
         避免模型用「交易多不多」作弊

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🔗 **管线中枢** —— 上承 data/raw/ 的原始交易，
        下接 src/modeling/plot_gonogo.py 的三分类与 go/no-go 判定。
        MODEL_FEATURES / LATENCY_ONLY_FEATURES 两个特征集供建模层直接引用。
'''

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils.common import log  # noqa: E402


# ================================================================
#  FeatureExtractor — 六大特征族统一封装
# ================================================================
class FeatureExtractor(object):
    """
    地址行为特征提取器

    核心函数 (共 3 个):
      ① extract(txs)                    单地址 → 35 个特征
      ② build_table(chain, synthetic)   批量 → 特征表 DataFrame
      ③ run(chain, synthetic)           ②+落盘+摘要

    六大特征族 (每族一个方法，便于消融实验按族剔除):
      f_rhythm / f_gas / f_nonce / f_interaction / f_value / f_latency

    静态统计量: burstiness / memory_coefficient / norm_entropy / band_fractions

    校验函数: validate_table

    Args:
        min_tx: 最低交易数门槛；缺省 config.MIN_TX_FOR_FEATURES
        bands:  (bot 带上界, agent 带上界)；缺省取 config.BAND_*
    """

    # ---- 参与建模的特征列（排除规模类，避免用「交易多不多」作弊）----
    MODEL_FEATURES = [
        'dt_p01', 'dt_p05', 'dt_p10', 'dt_p25', 'dt_p50', 'log_dt_median',
        'frac_lt2s', 'frac_2to10s', 'frac_gt10s',
        'burstiness', 'memory_coef', 'same_block_ratio', 'consec_block_ratio',
        'entropy_24h', 'entropy_dow', 'night_ratio', 'longest_silence_h',
        'gas_price_cv', 'gas_price_mode_share',
        'nonce_gap_ratio', 'unique_to_ratio', 'method_entropy',
        'failed_ratio', 'value_zero_ratio',
    ]

    # ---- 只用「延迟」这一族 —— 单独验证 H1 的判别力 ----
    LATENCY_ONLY_FEATURES = [
        'dt_p01', 'dt_p05', 'dt_p10', 'dt_p25', 'dt_p50', 'log_dt_median',
        'frac_lt2s', 'frac_2to10s', 'frac_gt10s',
    ]

    # 规模类特征（不进模型，仅供描述统计）
    SCALE_FEATURES = ['n_tx', 'span_days', 'tx_per_day']

    MIN_DT_SAMPLES = 5      # Δt 有效样本下限

    def __init__(self, min_tx: int = None, bands: tuple = None):
        """
        初始化

        Args:
            min_tx: 最低交易数门槛
            bands:  (bot 带上界, agent 带上界)
        Returns:
            None
        """
        self.min_tx = min_tx or config.MIN_TX_FOR_FEATURES
        self.bands = bands or (config.BAND_BOT[1], config.BAND_AGENT[1])
        self.n_skipped = 0
        self.n_cross_seg = 0

    # ============================================================
    #  通用工具 / 静态统计量
    # ============================================================
    @staticmethod
    def _safe_float(v, default: float = 0.0):
        """安全转 float: None/空/NaN/不可转 → default。"""
        try:
            if v is None or v == '':
                return default
            fv = float(v)
            return default if fv != fv else fv
        except (TypeError, ValueError):
            return default

    @staticmethod
    def burstiness(dt: np.ndarray) -> float:
        """
        Goh & Barabási (2008) 的 burstiness 系数

            B = (σ − μ) / (σ + μ)      ∈ [-1, 1]
            B = -1  完全周期（等间隔）  → 定时脚本
            B =  0  泊松（随机）
            B →  1  极度突发           → 事件驱动

        Args:
            dt: 相邻交易间隔数组
        Returns:
            float
        """
        if len(dt) < 2:
            return np.nan
        mu, sd = dt.mean(), dt.std()
        return (sd - mu) / (sd + mu) if (sd + mu) > 0 else np.nan

    @staticmethod
    def memory_coefficient(dt: np.ndarray) -> float:
        """
        相邻间隔的自相关（Goh & Barabási 的 memory coefficient M）

        M > 0 表示「短间隔后倾向再来短间隔」—— 成串的自动化行为。

        Args:
            dt: 相邻交易间隔数组
        Returns:
            float
        """
        if len(dt) < 3:
            return np.nan
        a, b = dt[:-1], dt[1:]
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def norm_entropy(counts: np.ndarray) -> float:
        """
        归一化 Shannon 熵，∈ [0,1]。1 = 完全均匀，0 = 全集中在一个桶。

        Args:
            counts: 各桶计数
        Returns:
            float
        """
        total = counts.sum()
        if total == 0:
            return np.nan
        p = counts[counts > 0] / total
        h = -(p * np.log(p)).sum()
        hmax = math.log(len(counts))
        return float(h / hmax) if hmax > 0 else 0.0

    def band_fractions(self, dt: np.ndarray) -> tuple:
        """
        三个假设频带各占多少 —— 最直观的 go/no-go 证据

        Args:
            dt: 相邻交易间隔数组
        Returns:
            (frac_lt2s, frac_2to10s, frac_gt10s)
        """
        if len(dt) == 0:
            return (np.nan,) * 3
        n, (b_bot, b_agent) = len(dt), self.bands
        return (float((dt < b_bot).sum() / n),
                float(((dt >= b_bot) & (dt < b_agent)).sum() / n),
                float((dt >= b_agent).sum() / n))

    @staticmethod
    def same_target_intervals(txs: list) -> np.ndarray:
        """
        M2：只统计对同一目标合约的连续交互间隔

        Args:
            txs: 已按时间升序的交易列表
        Returns:
            np.ndarray
        """
        out, last = [], {}
        for t in txs:
            to = t.get('to') or ''
            seg = t.get('seg', 0)
            key = (to, seg)          # ★ 同段同目标才配对，避免跨段假间隔
            if key in last:
                gap = t['timeStamp'] - last[key]
                if gap >= 0:
                    out.append(gap)
            last[key] = t['timeStamp']
        return np.array(out, dtype=float) if out else np.array([])

    # ============================================================
    #  六大特征族
    # ============================================================
    def f_latency(self, dt: np.ndarray, m2: np.ndarray) -> dict:
        """
        族 6 ★ 反应延迟 —— 本课题的核心创新特征

        Args:
            dt: M1 相邻交易间隔
            m2: M2 同目标响应间隔
        Returns:
            dict: 15 个延迟特征
        """
        dt_pos = dt[dt > 0]
        lt2, mid, gt10 = self.band_fractions(dt)
        return {
            'dt_min': float(dt.min()),
            'dt_p01': float(np.percentile(dt, 1)),
            'dt_p05': float(np.percentile(dt, 5)),
            'dt_p10': float(np.percentile(dt, 10)),
            'dt_p25': float(np.percentile(dt, 25)),
            'dt_p50': float(np.percentile(dt, 50)),
            'dt_p75': float(np.percentile(dt, 75)),
            'dt_mean': float(dt.mean()),
            'dt_std': float(dt.std()),
            'log_dt_median': float(np.median(np.log10(dt_pos))) if len(dt_pos) else np.nan,
            'frac_lt2s': lt2, 'frac_2to10s': mid, 'frac_gt10s': gt10,
            'm2_p05': float(np.percentile(m2, 5)) if len(m2) >= 5 else np.nan,
            'm2_p50': float(np.percentile(m2, 50)) if len(m2) >= 5 else np.nan,
            'm2_n': len(m2),
        }

    def f_rhythm(self, dt: np.ndarray, blocks: np.ndarray,
                 ts: np.ndarray) -> dict:
        """
        族 1 时序节律 —— 人类有昼夜节律，bot 全天均匀，agent 事件驱动

        Args:
            dt / blocks / ts: 间隔、区块号、时间戳数组
        Returns:
            dict: 8 个节律特征
        """
        hours = ((ts // 3600) % 24).astype(int)
        dows = ((ts // 86400 + 4) % 7).astype(int)   # 1970-01-01 是周四
        hour_hist = np.bincount(hours, minlength=24).astype(float)
        dow_hist = np.bincount(dows, minlength=7).astype(float)
        return {
            'burstiness': self.burstiness(dt),
            'memory_coef': self.memory_coefficient(dt),
            'same_block_ratio': float((dt == 0).sum() / len(dt)),
            'consec_block_ratio': float(
                (np.diff(blocks) == 1).sum() / max(len(blocks) - 1, 1)),
            'longest_silence_h': float(dt.max() / 3600),
            'entropy_24h': self.norm_entropy(hour_hist),
            'entropy_dow': self.norm_entropy(dow_hist),
            'night_ratio': float(hour_hist[0:6].sum() / hour_hist.sum()),
        }

    def f_gas(self, txs: list) -> dict:
        """
        族 2 Gas 策略 —— 已有论文证明钱包软件可被 gas 策略指纹化

        Args:
            txs: 交易列表
        Returns:
            dict: 2 个 gas 特征
        """
        gp = np.array([self._safe_float(t.get('gasPrice')) for t in txs])
        nz = gp[gp > 0]
        cv = (float(nz.std() / nz.mean())
              if len(nz) > 1 and nz.mean() > 0 else np.nan)
        mode_share = (float(Counter(nz.tolist()).most_common(1)[0][1] / len(nz))
                      if len(nz) else np.nan)
        return {'gas_price_cv': cv, 'gas_price_mode_share': mode_share}

    def f_nonce(self, txs: list) -> dict:
        """
        族 3 Nonce —— 并发 pending nonce 是程序化操作的强信号

        Args:
            txs: 交易列表
        Returns:
            dict: 1 个 nonce 特征
        """
        nonces = np.array([self._safe_float(t.get('nonce')) for t in txs])
        dn = np.diff(nonces)
        return {'nonce_gap_ratio':
                float((dn > 1).sum() / len(dn)) if len(dn) else np.nan}

    def f_interaction(self, txs: list) -> dict:
        """
        族 4 交互结构 —— 脚本高重复，人类高多样

        Args:
            txs: 交易列表
        Returns:
            dict: 3 个交互特征
        """
        tos = [t.get('to') or '' for t in txs]
        methods = [t.get('methodId') or '0x' for t in txs]
        return {
            'unique_to_ratio': len(set(tos)) / len(tos),
            'method_entropy': self.norm_entropy(
                np.array(list(Counter(methods).values()), dtype=float)),
            'failed_ratio': float(
                sum(1 for t in txs if t.get('isError') == '1') / len(txs)),
        }

    def f_value(self, txs: list) -> dict:
        """
        族 5 金额偏好 —— 整数偏好是脚本特征

        Args:
            txs: 交易列表
        Returns:
            dict: 2 个金额特征
        """
        vals = np.array([self._safe_float(t.get('value')) for t in txs])
        nz = vals[vals > 0]
        return {
            'value_zero_ratio': float((vals == 0).sum() / len(vals)),
            # 「整数偏好」：金额是 0.01 ETH 的整数倍
            'round_value_ratio':
                float((nz % 1e16 == 0).sum() / len(nz)) if len(nz) else np.nan,
        }

    # ============================================================
    #  ① 单地址特征提取
    # ============================================================
    def extract(self, txs: list) -> dict:
        """
        ① 从一个地址的交易列表算全部特征

        【数据源】 data/raw/{chain}/{group}/{address}.json
                 由 src/data_fetch/fetch_txs.py 产出（Etherscan / SQD 双后端已统一字段）

        【指标含义】 一句话: "把一串交易压缩成 35 维行为指纹"
                    六大族分别刻画：何时做(节律)、多快做(延迟)、
                    怎么出价(gas)、怎么排队(nonce)、跟谁交互、金额偏好。

        【返回字段】 35 个特征 + 3 个规模类（n_tx / span_days / tx_per_day）
                    详见模块文档 二

        【阈值解读】
                    · 交易数 < 20      → 统计量不可靠，返回 {}
                    · Δt 有效样本 < 5  → 返回 {}
                    · burstiness ≈ -1  → 定时脚本
                    · gas_price_mode_share ≈ 1 → gasPrice 恒定 = 硬编码脚本

        【如何使用】
                   1) 建模只用 MODEL_FEATURES（已排除规模类，避免作弊）
                   2) 消融实验按族剔除：调 f_gas / f_nonce 等单族方法
                   3) 只验证 H1 时用 LATENCY_ONLY_FEATURES

        Args:
            txs: 该地址的交易列表
        Returns:
            dict: 特征字典；不满足门槛时返回 {}
        """
        if len(txs) < self.min_tx:          # 1. 交易数门槛
            self.n_skipped += 1
            return {}

        # 2. 按 (timeStamp, txIndex) 升序
        txs = sorted(txs, key=lambda t: (t['timeStamp'], t.get('txIndex', 0)))
        ts = np.array([t['timeStamp'] for t in txs], dtype=float)
        blocks = np.array([t['blockNumber'] for t in txs], dtype=float)

        # ★ 分段封顶的地址带 seg 字段 —— Δt 只在段内算，跳过人为的跨段断点。
        #   （fetch_activity 对高频地址按 12 段各取 N 条，跨段间隔是采样造成的，
        #     不是真实的行为间隔，必须剔除，否则会污染延迟特征）
        segs = np.array([t.get('seg', 0) for t in txs], dtype=float)
        d_all = np.diff(ts)
        same_seg = np.diff(segs) == 0
        dt = d_all[same_seg & (d_all >= 0)]
        self.n_cross_seg += int((~same_seg).sum())

        if len(dt) < self.MIN_DT_SAMPLES:   # 3. Δt 样本门槛
            self.n_skipped += 1
            return {}

        span_s = float(ts[-1] - ts[0])
        feat = {
            'n_tx': len(txs),
            'span_days': span_s / 86400,
            'tx_per_day': len(txs) / max(span_s / 86400, 1e-9),
        }
        feat.update(self.f_latency(dt, self.same_target_intervals(txs)))
        feat.update(self.f_rhythm(dt, blocks, ts))
        feat.update(self.f_gas(txs))
        feat.update(self.f_nonce(txs))
        feat.update(self.f_interaction(txs))
        feat.update(self.f_value(txs))
        return feat

    # ============================================================
    #  ② 批量构表
    # ============================================================
    def build_table(self, chain: str = None,
                    synthetic: bool = False) -> pd.DataFrame:
        """
        ② 遍历三组地址的原始交易，构建特征表

        Args:
            chain:     链名；缺省 config.PRIMARY_CHAIN
            synthetic: True 时读 data/raw/synthetic/
        Returns:
            pd.DataFrame: [address, group, ...特征]
        """
        base = config.DATA_RAW / ('synthetic' if synthetic
                                  else (chain or config.PRIMARY_CHAIN))
        if not base.exists():
            log(f'找不到原始数据目录 {base}', 'err')
            log('先跑：python3 src/utils/make_synthetic.py' if synthetic
                else '先跑：python3 src/data_fetch/fetch_txs.py', 'info')
            sys.exit(1)

        self.n_skipped = self.n_cross_seg = 0
        rows = []
        for group in config.GROUPS:
            gdir = base / group
            if not gdir.exists():
                log(f'[{group}] 无数据目录，跳过', 'warn')
                continue
            files = sorted(gdir.glob('*.json'))
            n_ok = 0
            for fp in files:
                try:
                    txs = json.loads(fp.read_text())
                except json.JSONDecodeError:
                    continue
                feat = self.extract(txs)
                if not feat:
                    continue
                feat['address'] = fp.stem
                feat['group'] = group
                rows.append(feat)
                n_ok += 1
            log(f'[{group}] {n_ok}/{len(files)} 个地址通过 ≥{self.min_tx} 笔交易的门槛',
                'info')

        if not rows:
            log('没有地址通过门槛。检查 MIN_TX_FOR_FEATURES 或数据质量。', 'err')
            sys.exit(1)

        df = pd.DataFrame(rows)
        cols = ['address', 'group'] + [c for c in df.columns
                                       if c not in ('address', 'group')]
        return df[cols]

    # ============================================================
    #  ③ 落盘与摘要
    # ============================================================
    def save(self, df: pd.DataFrame, tag: str) -> Path:
        """写入 data/features/features_{tag}.csv。"""
        out = config.DATA_FEAT / f'features_{tag}.csv'
        df.to_csv(out, index=False)
        log(f'写入 {out.relative_to(config.ROOT)}  '
            f'({len(df)} 行 × {len(df.columns)} 列)', 'ok')
        return out

    def summarize(self, df: pd.DataFrame) -> None:
        """打印分组样本量与延迟核心指标的中位数。"""
        print()
        print('各组样本量：')
        print(df['group'].value_counts().to_string())
        print()
        print('★ 延迟核心指标的分组中位数：')
        key = ['dt_p05', 'dt_p50', 'frac_lt2s', 'frac_2to10s',
               'frac_gt10s', 'burstiness']
        print(df.groupby('group')[key].median().round(3).to_string())
        print()

    def run(self, chain: str = None, synthetic: bool = False) -> pd.DataFrame:
        """
        ③ 完整流程：构表 → 落盘 → 摘要 → 校验

        Args:
            chain / synthetic
        Returns:
            pd.DataFrame
        """
        tag = 'synthetic' if synthetic else (chain or config.PRIMARY_CHAIN)
        log(f'计算特征：{tag}', 'step')
        df = self.build_table(chain, synthetic)
        self.save(df, tag)
        self.summarize(df)
        self.validate_table(df)
        log('下一步：python3 src/modeling/plot_gonogo.py'
            + (' --synthetic' if synthetic else ''), 'info')
        return df

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

    def validate_table(self, df: pd.DataFrame,
                       context: str = '特征表') -> bool:
        """
        校验特征表

        检查项:
          1. 非空 / 建模特征列齐全
          2. 地址无重复
          3. 比例类特征落在 [0,1]
          4. burstiness 落在 [-1,1]
          5. ⚠️ 全 NaN 的特征列
          6. ⚠️ 分组样本量不足或严重不平衡
          7. ⚠️ 三组是否齐（三分类需要三组）
          8. 🔴 分辨率天花板提示

        Args:
            df / context
        Returns:
            bool
        """
        issues, warns = [], []

        if df is None or df.empty:
            issues.append('特征表为空')
            return self._emit(context, issues, warns)

        miss = [c for c in self.MODEL_FEATURES if c not in df.columns]
        if miss:
            issues.append(f'缺建模特征列: {miss}')
        if 'group' not in df.columns or 'address' not in df.columns:
            issues.append('缺 address / group 列')
        if issues:
            return self._emit(context, issues, warns)

        dup = int(df['address'].duplicated().sum())
        if dup:
            issues.append(f'{dup} 个地址重复')

        # 3. 比例类特征范围
        for c in ('frac_lt2s', 'frac_2to10s', 'frac_gt10s', 'same_block_ratio',
                  'consec_block_ratio', 'entropy_24h', 'entropy_dow',
                  'night_ratio', 'unique_to_ratio', 'method_entropy',
                  'failed_ratio', 'value_zero_ratio'):
            if c in df.columns:
                s = df[c].dropna()
                if len(s) and (s.min() < -1e-9 or s.max() > 1 + 1e-9):
                    issues.append(f'{c} 超出 [0,1]：[{s.min():.3f}, {s.max():.3f}]')

        # 4. burstiness 范围
        if 'burstiness' in df.columns:
            s = df['burstiness'].dropna()
            if len(s) and (s.min() < -1 - 1e-9 or s.max() > 1 + 1e-9):
                issues.append(f'burstiness 超出 [-1,1]：[{s.min():.3f}, {s.max():.3f}]')

        # 5. 全 NaN 列
        allnan = [c for c in self.MODEL_FEATURES if df[c].isna().all()]
        if allnan:
            warns.append(f'{len(allnan)} 个特征全为 NaN: {allnan[:5]} '
                         f'—— 上游字段可能缺失')

        # 6/7. 分组情况
        vc = df['group'].value_counts()
        missing_g = set(config.GROUPS) - set(vc.index)
        if missing_g:
            warns.append(f'缺少分组 {sorted(missing_g)} —— '
                         f'三分类需要三组齐全')
        for g, n in vc.items():
            if n < 30:
                warns.append(f'{g} 组仅 {n} 个样本，交叉验证会不稳定')
        if len(vc) > 1 and vc.max() / vc.min() > 5:
            warns.append(f'类别严重不平衡 {dict(vc)} —— '
                         f'训练需 class_weight，评估看 macro-F1')

        if self.n_skipped:
            warns.append(f'{self.n_skipped} 个地址因交易数不足被跳过')
        if self.n_cross_seg:
            warns.append(f'{self.n_cross_seg} 个跨段断点已从 Δt 中剔除'
                         f'（分段封顶造成的人为间隔，不是真实行为）')

        # 8. 分辨率天花板
        if 'frac_2to10s' in df.columns:
            n_full = int((df['frac_2to10s'] > 0.99).sum())
            if n_full:
                warns.append(f'{n_full} 个地址 frac_2to10s > 0.99 —— '
                             f'⚠️ Base 出块 2s 正好压在 agent 带下边界，'
                             f'纯 bot 也会命中，是已知假阳性')

        warns.append('🔴 提醒：Δt 是代理量不是真正的 event-response latency；'
                     '亚秒差异在链上永远不可见（时间戳整秒）')
        return self._emit(context, issues, warns)


# ---- 向后兼容：plot_gonogo.py 直接 import 这两个常量 ----
MODEL_FEATURES = FeatureExtractor.MODEL_FEATURES
LATENCY_ONLY_FEATURES = FeatureExtractor.LATENCY_ONLY_FEATURES


def main():
    ap = argparse.ArgumentParser(description='行为特征工程（六大特征族）')
    ap.add_argument('--chain', default=config.PRIMARY_CHAIN, choices=list(config.CHAINS))
    ap.add_argument('--synthetic', action='store_true', help='使用合成数据')
    ap.add_argument('--min-tx', type=int, default=None, help='覆盖最低交易数门槛')
    args = ap.parse_args()
    FeatureExtractor(min_tx=args.min_tx).run(chain=args.chain,
                                             synthetic=args.synthetic)


if __name__ == '__main__':
    main()
