#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     : agent_fingerprint
@File        : f02_rhythm.py
@Description :
    ============================================================
    一、特征体系 (业务场景分类)
    ============================================================
      类别 2/6:「时序节律」(12 特征) —— 回答: 这个地址「什么时候做事」?
      课题依据: 三类主体的作息形态本质不同
        人类      : 有昼夜节律（要睡觉）⇒ 24h 熵低、有连续静默段、夜间占比低
        传统 bot  : 7×24 全天均匀 ⇒ 24h 熵高、burstiness 接近 -1（定时轮询）
        AI agent  : 事件驱动 ⇒ 24h 熵中等偏高、burstiness 高（成串爆发）

      ⚠️ 🔴 循环风险（文档反复警告的坑）:
        若用「昼夜节律」筛选人类样本、又用 entropy_24h 当特征
        = 用答案筛样本，会人为抬高模型性能。
        ⇒ 本课题的 HumanDataSource **绝不用节律筛选**（只用 CEX 提币 + EOA 两条判据），
          因此 entropy_24h / night_ratio 是**独立证据**，可放心进模型。
        ⇒ 但仍建议做消融: 剔除本族后性能掉多少，把「筛选偏差有多大」写成论文一节。

    ============================================================
    二、特征定义 (逻辑 / 数据源 / 业务意义 / 特征生成时间)
    ============================================================
      数据源: 统一活动流的 timeStamp / blockNumber / seg

      | 字段                  | 计算逻辑                        | 业务意义                    | 生成时间 |
      |-----------------------|---------------------------------|-----------------------------|----------|
      | f_burstiness          | (σ-μ)/(σ+μ) of Δt ∈[-1,1]       | -1周期(定时脚本)/→1事件驱动 | 活动落块后 |
      | f_memory_coef         | 相邻 Δt 的自相关                | >0 短间隔成串 = 自动化      | 同上     |
      | f_same_block_ratio    | Δt==0 的占比                    | ★同区块多笔 = bot 强信号    | 同上     |
      | f_consec_block_ratio  | 区块号差==1 的占比              | ★连续区块 = 抢排序          | 同上     |
      | f_longest_silence_h   | 最大 Δt / 本组窗口秒数          | 最长静默段（人类睡眠）      | 同上     |
      | f_entropy_24h         | 小时分布的归一化 Shannon 熵     | ★1=全天均匀 / 低=有作息     | 同上     |
      | f_entropy_dow         | 周内分布的归一化熵              | 工作日/周末差异             | 同上     |
      | f_night_ratio         | UTC 0-6 点活动占比              | ★人类通常显著偏低           | 同上     |
      | f_peak_hour_share     | 最活跃小时的占比                | 集中度                      | 同上     |
      | f_active_hours        | 有活动的不同小时数 (0~24)       | 覆盖广度                    | 同上     |
      | f_active_days         | 活跃日数 / 窗口天数             | 持续性                      | 同上     |
      | f_span_days           | 首末跨度 / 窗口跨度             | 覆盖度（0~1）               | 同上     |

      🔴 末三个特征已做「窗口归一化」: 各组采集窗口不等长（human ×12，见
         config.GROUP_WINDOW_MULT），绝对跨度会直接泄漏组别。除以本组窗口后，
         它们表达的是「占窗口的比例」而非「绝对多久」，三组重新可比。
         这三个特征在 f_ensemble 中登记为 WINDOW_DEP_FEATURES 供消融对照。

    ============================================================
    三、特征调用方式
    ============================================================
      from src.feature.f02_rhythm import FeatureRhythm

      Ins = FeatureRhythm()
      feat = Ins.get_feature(acts)   # → {'f_burstiness': 0.1, 'f_entropy_24h': 0.95, ...}

    ============================================================
    四、使用注意事项
    ============================================================
      1) 熵是**时区无关**的（换算时区只是重排直方图的桶），可放心跨地域用。
         但 f_night_ratio 依赖 UTC 0-6 的定义，跨时区人群会有偏差 —— 论文需说明。
      2) entropy_dow 需要 ≥7 天跨度才有意义。fetch_activity 的分段封顶正是为了
         保住跨度（若取「最后 N 条」，高频地址的跨度会被压到几小时，本族全废）。
      3) burstiness 与 f01 的 Δt 共用同一段内间隔，两族有相关性，
         树模型无妨；线性模型需注意共线性。

    Update:
'''

import math

import numpy as np


# ================================================================
#  FeatureRhythm — 时序节律 (12 特征)
# ================================================================
class FeatureRhythm(object):
    """
    时序节律特征类。回答「这个地址什么时候做事」。

    Args:
        night_hours: 定义为「夜间」的 UTC 小时区间，缺省 (0, 6)
    """

    def __init__(self, night_hours: tuple = (0, 6), block_time: float = 2.0):
        """
        构造: 存夜间小时区间 + 出块时间（把 win_lo/win_hi 的块数换算成秒）。

        Args:
            night_hours: 夜间小时区间，缺省 (0,6) 即 00:00~06:00
            block_time:  出块秒数，缺省 2.0（Base 实测）；
                         调用方（f_ensemble）按链传 config.SQD_BLOCK_TIME_MEASURED
        """
        self.night_hours = night_hours
        self.block_time = float(block_time)

    # ============================================================
    #  burstiness() — Goh & Barabási (2008) 突发性系数
    # ============================================================
    @staticmethod
    def burstiness(dt: np.ndarray) -> float:
        """
        B = (σ − μ) / (σ + μ) ∈ [-1, 1]

        B = -1  完全周期（等间隔）→ 定时脚本
        B =  0  泊松（随机）
        B →  1  极度突发         → 事件驱动

        Args:
            dt: 段内间隔数组
        Returns:
            float
        """
        if len(dt) < 2:
            return np.nan
        mu, sd = dt.mean(), dt.std()
        return (sd - mu) / (sd + mu) if (sd + mu) > 0 else np.nan

    # ============================================================
    #  memory_coefficient() — 相邻间隔自相关
    # ============================================================
    @staticmethod
    def memory_coefficient(dt: np.ndarray) -> float:
        """
        M > 0 表示「短间隔后倾向再来短间隔」—— 成串的自动化行为。

        Args:
            dt: 段内间隔数组
        Returns:
            float
        """
        if len(dt) < 3:
            return np.nan
        a, b = dt[:-1], dt[1:]
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    # ============================================================
    #  norm_entropy() — 归一化 Shannon 熵
    # ============================================================
    @staticmethod
    def norm_entropy(counts: np.ndarray) -> float:
        """
        归一化到 [0,1]。1 = 完全均匀，0 = 全集中在一个桶。

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

    # ============================================================
    #  get_feature() — 装配全部节律特征
    # ============================================================
    def get_feature(self, acts: list) -> dict:
        """
        计算该地址的全部时序节律特征

        Args:
            acts: 该地址的活动流，需已按时间升序
        Returns:
            dict: 12 个特征；样本不足时返回 {}
        """
        if len(acts) < 2:
            return {}
        ts = np.array([a['timeStamp'] for a in acts], dtype=float)
        blocks = np.array([a['blockNumber'] for a in acts], dtype=float)
        seg = np.array([a.get('seg', 0) for a in acts], dtype=float)

        d = np.diff(ts)
        same = np.diff(seg) == 0
        dt = d[same & (d >= 0)]
        if len(dt) < 2:
            return {}

        hours = ((ts // 3600) % 24).astype(int)
        dows = ((ts // 86400 + 4) % 7).astype(int)   # 1970-01-01 是周四
        hour_hist = np.bincount(hours, minlength=24).astype(float)
        dow_hist = np.bincount(dows, minlength=7).astype(float)
        n0, n1 = self.night_hours

        db = np.diff(blocks)
        span_s = float(ts[-1] - ts[0])

        # 🔴 窗口归一化 —— 各组采集窗口不等长（human ×12，见 config.GROUP_WINDOW_MULT）。
        #    若直接输出绝对跨度，模型只要学「跨度 > 1 天 ⇒ human」就能作弊，
        #    这与行为无关，纯粹是采集设计泄漏。除以本组窗口长度后三组重新可比。
        w_lo = acts[0].get('win_lo')
        w_hi = acts[0].get('win_hi')
        if w_lo is not None and w_hi is not None and w_hi > w_lo:
            win_s = (w_hi - w_lo) * self.block_time     # 本组窗口秒数
        else:
            win_s = max(span_s, 1.0)                    # 老数据无 win_* 时退化为自身跨度
        win_s = max(win_s, 1.0)

        return {
            'f_burstiness': self.burstiness(dt),
            'f_memory_coef': self.memory_coefficient(dt),
            'f_same_block_ratio': float((dt == 0).sum() / len(dt)),
            'f_consec_block_ratio': float((db[same] == 1).sum() / max(same.sum(), 1)),
            # ↓ 三个跨度特征全部归一化为「占本组窗口的比例」，量纲统一到 [0,1]
            'f_longest_silence_h': float(dt.max() / win_s),
            'f_entropy_24h': self.norm_entropy(hour_hist),
            'f_entropy_dow': self.norm_entropy(dow_hist),
            'f_night_ratio': float(hour_hist[n0:n1].sum() / max(hour_hist.sum(), 1)),
            'f_peak_hour_share': float(hour_hist.max() / max(hour_hist.sum(), 1)),
            'f_active_hours': int((hour_hist > 0).sum()),
            'f_active_days': float(len(set((ts // 86400).astype(int)))
                                   / max(win_s / 86400, 1.0)),
            'f_span_days': span_s / win_s,
        }
