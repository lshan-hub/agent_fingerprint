#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     : agent_fingerprint
@File        : f01_latency.py
@Description :
    ============================================================
    一、特征体系 (业务场景分类)
    ============================================================
      类别 1/6:「反应延迟」(18 特征) —— 回答: 这个地址「最快能多快做下一个决策」?
      课题依据: 核心假设 H1 —— LLM 推理有物理下界，构成 bot 与人类都不占据的频带
        传统 bot  : 亚秒 ~ 2 秒   （预编译 if-else，还要抢区块排序）
        AI agent  : 2 ~ 10 秒     （★ LLM 前向传播 + API 网络往返，压不下去）
        人类       : >10 秒，方差极大（会走神、会犹豫）

      🎯 为什么这一族最值得下注:
        ① 对应真实物理约束，不是统计巧合
        ② 难以规避 —— 可伪造 gas 策略，但没法让 LLM 推理变快；
           人为加延迟就丧失了 agent 抢先执行的经济优势（evasion-cost 论证）
        ③ 正面回应「链上特征与 Web 域实质不同在哪」:
           Web 域靠 Playwright 的实现缺陷，链上靠推理成本的物理下界，后者更本质

    ============================================================
    二、特征定义 (逻辑 / 数据源 / 业务意义 / 特征生成时间)
    ============================================================
      数据源: data/raw/{chain}/{group}/{address}.json 的统一活动流
              （由 src/data_fetch/fetch_activity.py 产出）
              必需字段: timeStamp / blockNumber / txIndex / seg / to

      | 字段            | 计算逻辑                      | 业务意义                       | 生成时间 |
      |-----------------|-------------------------------|--------------------------------|----------|
      | f_dt_min        | 段内相邻活动间隔的最小值      | 理论最快响应                   | 活动落块后 |
      | f_dt_p01/05/10  | 间隔的 1/5/10 分位            | ★最快响应能力（主判别量）      | 同上     |
      | f_dt_p25/50/75  | 间隔的 25/50/75 分位          | 典型响应节奏                   | 同上     |
      | f_dt_mean/std   | 间隔均值/标准差               | 中心与离散                     | 同上     |
      | f_log_dt_median | log10(间隔>0) 的中位数        | 对数尺度中心（跨量级可比）     | 同上     |
      | f_frac_lt2s     | 间隔 <2s 的占比               | ★落在 bot 带的比例             | 同上     |
      | f_frac_2to10s   | 间隔 ∈[2,10)s 的占比          | ★落在 agent 带的比例           | 同上     |
      | f_frac_gt10s    | 间隔 ≥10s 的占比              | ★落在 human 带的比例           | 同上     |
      | f_m2_p05/p50    | 同目标响应间隔的 5/50 分位    | 「观察→决策→行动」循环节奏     | 同上     |
      | f_m2_n          | 同目标配对样本数              | M2 的可信度                    | 同上     |
      | f_dt_iqr        | p75 - p25                     | 稳健离散度（抗极端值）         | 同上     |
      | f_dt_cv         | std / mean                    | 变异系数（无量纲）             | 同上     |

    ============================================================
    三、特征调用方式
    ============================================================
      from src.feature.f01_latency import FeatureLatency

      Ins = FeatureLatency()
      feat = Ins.get_feature(acts)      # acts = 该地址的活动流 list[dict]
      # → {'f_dt_p05': 0.0, 'f_frac_2to10s': 0.5, ...}

    ============================================================
    四、使用注意事项
    ============================================================
      1) 🔴 Δt 只在 **同一 seg 内**计算。采集端对高频组分段抽样
         （bot 组为 config.BOT_COLLECT_SEGMENTS=36 段 × 10,000 块；
         agent/human 全窗口连续，seg 恒为 0），跨段间隔是采样造成的
         人为断点，不是真实行为间隔，必须剔除。
      🔴 由此带来一处必须在论文里交代的不对称：bot 组段内 Δt 的物理上限
         约为 10,000 块 × 2s = 20,000 秒，而 agent/human 的上限是整个窗口。
         延迟族的尾部统计量（p75/mean/std/iqr）因此组间不完全可比 ——
         定量影响见 src/verify/robustness_checks.py 的 Δt 截断对照（表 B）。
      2) 🔴 分辨率天花板: EVM 区块时间戳是整秒，Base 出块 2s
         ⇒ Δt 物理上只能取 {0, 2, 4, ...}，亚秒差异永远不可见。
         且 Δt=2s（连续区块）正好压在 agent 带下边界 —— 纯 bot 也会命中，是已知假阳性。
         对策: 配合 f02_rhythm 的 same_block_ratio / consec_block_ratio 使用。
      3) Δt 可以合法地等于 0（同秒多笔 = bot 典型形态），
         做 log 变换时必须 clip 而非丢弃，否则会删掉 bot 组的主体信号。
      4) 有效样本 < MIN_DT_SAMPLES(5) 时返回空 dict，调用方需处理。

    Update:
'''

import numpy as np


# ================================================================
#  FeatureLatency — 反应延迟 (18 特征)
# ================================================================
class FeatureLatency(object):
    """
    反应延迟特征类，课题核心创新特征族。

    Args:
        bands: (bot 带上界, agent 带上界)，缺省 (2.0, 10.0)
    """

    MIN_DT_SAMPLES = 5      # Δt 有效样本下限
    LOG_FLOOR = 0.1         # log 变换的下限（Δt=0 clip 到此值，不丢弃）

    def __init__(self, bands: tuple = (2.0, 10.0)):
        """构造: 存三频带边界。"""
        self.bands = bands

    # ============================================================
    #  seg_deltas() — 段内相邻活动间隔（跨段断点已剔除）
    # ============================================================
    @staticmethod
    def seg_deltas(acts: list) -> np.ndarray:
        """
        计算段内相邻活动间隔

        🔴 只在同一 seg 内算 —— 跨段间隔是分段封顶造成的人为断点。

        Args:
            acts: 已按时间升序的活动流
        Returns:
            np.ndarray: 段内间隔（秒），非负
        """
        if len(acts) < 2:
            return np.array([])
        ts = np.array([a['timeStamp'] for a in acts], dtype=float)
        seg = np.array([a.get('seg', 0) for a in acts], dtype=float)
        d = np.diff(ts)
        same = np.diff(seg) == 0
        return d[same & (d >= 0)]

    # ============================================================
    #  same_target_deltas() — M2 同目标响应间隔
    # ============================================================
    @staticmethod
    def same_target_deltas(acts: list) -> np.ndarray:
        """
        只统计对同一目标合约(to)的连续交互间隔

        语义上更接近「观察 → 决策 → 行动」的循环，噪声更低但样本更少。
        同样按 (to, seg) 配对，避免跨段假间隔。

        Args:
            acts: 已按时间升序的活动流
        Returns:
            np.ndarray
        """
        out, last = [], {}
        for a in acts:
            key = (a.get('to') or '', a.get('seg', 0))
            if key in last:
                gap = a['timeStamp'] - last[key]
                if gap >= 0:
                    out.append(gap)
            last[key] = a['timeStamp']
        return np.array(out, dtype=float) if out else np.array([])

    # ============================================================
    #  band_fractions() — 三频带占比，最直观的 H1 证据
    # ============================================================
    def band_fractions(self, dt: np.ndarray) -> tuple:
        """
        三个假设频带各占多少

        Args:
            dt: 段内间隔数组
        Returns:
            (frac_lt2s, frac_2to10s, frac_gt10s)
        """
        if len(dt) == 0:
            return (np.nan,) * 3
        n, (b1, b2) = len(dt), self.bands
        return (float((dt < b1).sum() / n),
                float(((dt >= b1) & (dt < b2)).sum() / n),
                float((dt >= b2).sum() / n))

    # ============================================================
    #  get_feature() — 装配全部延迟特征
    # ============================================================
    def get_feature(self, acts: list) -> dict:
        """
        计算该地址的全部反应延迟特征

        Args:
            acts: 该地址的活动流 list[dict]，需已按 (timeStamp, txIndex) 升序
        Returns:
            dict: 18 个特征；有效样本不足时返回 {}
        """
        dt = self.seg_deltas(acts)
        if len(dt) < self.MIN_DT_SAMPLES:
            return {}
        m2 = self.same_target_deltas(acts)
        pos = dt[dt > 0]
        lt2, mid, gt10 = self.band_fractions(dt)
        p25, p75 = float(np.percentile(dt, 25)), float(np.percentile(dt, 75))
        mean, std = float(dt.mean()), float(dt.std())

        return {
            'f_dt_min': float(dt.min()),
            'f_dt_p01': float(np.percentile(dt, 1)),
            'f_dt_p05': float(np.percentile(dt, 5)),
            'f_dt_p10': float(np.percentile(dt, 10)),
            'f_dt_p25': p25,
            'f_dt_p50': float(np.percentile(dt, 50)),
            'f_dt_p75': p75,
            'f_dt_mean': mean,
            'f_dt_std': std,
            'f_dt_iqr': p75 - p25,
            'f_dt_cv': std / mean if mean > 0 else np.nan,
            'f_log_dt_median': float(np.median(np.log10(pos))) if len(pos) else np.nan,
            'f_frac_lt2s': lt2,
            'f_frac_2to10s': mid,
            'f_frac_gt10s': gt10,
            'f_m2_p05': float(np.percentile(m2, 5)) if len(m2) >= 5 else np.nan,
            'f_m2_p50': float(np.percentile(m2, 50)) if len(m2) >= 5 else np.nan,
            'f_m2_n': len(m2),
        }
