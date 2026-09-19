#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     : agent_fingerprint
@File        : f03_gas.py
@Description :
    ============================================================
    一、特征体系 (业务场景分类)
    ============================================================
      类别 3/6:「Gas 策略指纹」(7 特征) —— 回答: 这个地址「怎么出价」?
      课题依据: priorityFee 完全由发送方软件决定，已有论文证明
        **钱包软件可被 gas 策略指纹化**。
        MetaMask 默认档 → 一类分布 / 硬编码脚本 → gasPrice 恒定 /
        MEV bot → 随竞争动态调整，方差极大

      ⚠️ 特征可得性差异（必须在论文交代）:
        只有 kind='tx_from' 的活动才有真实 gasPrice —— 那是该地址自己付的 gas。
        · MEV bot / 人类  : tx_from 占比高 ⇒ 本族特征有效
        · x402 付款方      : 走元交易，Facilitator 代付 ⇒ 本族特征几乎全缺失
        · Olas Safe        : 合约账户被调用 ⇒ 同样缺失
        ⇒ 本族对 agent 组是**结构性缺失**，不是随机缺失。
          这本身就是一个判别信号（缺失模式即信息），但不能当作「行为差异」解读。

    ============================================================
    二、特征定义 (逻辑 / 数据源 / 业务意义 / 特征生成时间)
    ============================================================
      数据源: 统一活动流的 gasPrice / kind

      | 字段                    | 计算逻辑                    | 业务意义                  | 生成时间 |
      |-------------------------|-----------------------------|---------------------------|----------|
      | f_gas_price_cv          | std/mean of gasPrice>0      | ★恒定=硬编码脚本          | 活动落块后 |
      | f_gas_price_mode_share  | 众数占比                    | ★接近1=完全硬编码         | 同上     |
      | f_gas_price_median      | 中位数（Gwei）              | 出价档位                  | 同上     |
      | f_gas_price_p95_p50     | p95/p50 比值                | 极端加价倾向（抢排序）    | 同上     |
      | f_gas_nunique_ratio     | 不同取值数/样本数           | 出价多样性                | 同上     |
      | f_gas_avail_ratio       | gasPrice>0 的活动占比       | ★特征可得性（缺失模式）   | 同上     |
      | f_txfrom_ratio          | kind=='tx_from' 的占比      | ★活动形态（自发 vs 被动） | 同上     |

    ============================================================
    三、特征调用方式
    ============================================================
      from src.feature.f03_gas import FeatureGas

      Ins = FeatureGas()
      feat = Ins.get_feature(acts)   # → {'f_gas_price_cv': 0.02, ...}

    ============================================================
    四、使用注意事项
    ============================================================
      1) gasPrice 单位是 wei，f_gas_price_median 已换算成 Gwei 便于阅读。
      2) 无有效 gasPrice 时各项返回 NaN（不是 0）——
         下游 SimpleImputer 会按中位数填充，不要在此处填 0 造成假信号。
      3) f_gas_avail_ratio 与 f_txfrom_ratio 是**元特征**（描述数据可得性本身），
         它们对 agent 组有极强判别力，但要警惕:
         这反映的是「协议形态差异」而非「行为差异」，论文中需分开讨论。

    Update:
'''

import numpy as np
from collections import Counter


# ================================================================
#  FeatureGas — Gas 策略指纹 (7 特征)
# ================================================================
class FeatureGas(object):
    """Gas 出价策略特征类。回答「这个地址怎么出价」。"""

    GWEI = 1e9

    def __init__(self):
        """无状态特征类，构造无需参数。"""
        pass

    @staticmethod
    def _f(v, default=0.0):
        """安全转 float。"""
        try:
            if v is None or v == '':
                return default
            fv = float(v)
            return default if fv != fv else fv
        except (TypeError, ValueError):
            return default

    # ============================================================
    #  get_feature() — 装配全部 Gas 特征
    # ============================================================
    def get_feature(self, acts: list) -> dict:
        """
        计算该地址的 Gas 策略特征

        Args:
            acts: 该地址的活动流
        Returns:
            dict: 7 个特征
        """
        if not acts:
            return {}
        gp = np.array([self._f(a.get('gasPrice')) for a in acts])
        nz = gp[gp > 0]
        n_from = sum(1 for a in acts if a.get('kind') == 'tx_from')

        out = {
            'f_gas_avail_ratio': float(len(nz) / len(acts)),
            'f_txfrom_ratio': float(n_from / len(acts)),
        }
        if len(nz) == 0:
            out.update({k: np.nan for k in
                        ('f_gas_price_cv', 'f_gas_price_mode_share',
                         'f_gas_price_median', 'f_gas_price_p95_p50',
                         'f_gas_nunique_ratio')})
            return out

        mean = nz.mean()
        p50, p95 = float(np.percentile(nz, 50)), float(np.percentile(nz, 95))
        out.update({
            'f_gas_price_cv': float(nz.std() / mean) if mean > 0 else np.nan,
            'f_gas_price_mode_share':
                float(Counter(nz.tolist()).most_common(1)[0][1] / len(nz)),
            'f_gas_price_median': p50 / self.GWEI,
            'f_gas_price_p95_p50': p95 / p50 if p50 > 0 else np.nan,
            'f_gas_nunique_ratio': float(len(set(nz.tolist())) / len(nz)),
        })
        return out
