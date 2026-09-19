#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     : agent_fingerprint
@File        : f04_nonce.py
@Description :
    ============================================================
    一、特征体系 (业务场景分类)
    ============================================================
      类别 4/6:「Nonce 并发」(5 特征) —— 回答: 这个地址「怎么排队发交易」?
      课题依据: nonce 是账户的交易计数器，必须严格递增且不跳号。
        · 人类用钱包: 一笔确认了再发下一笔 ⇒ nonce 顺序、无并发
        · 脚本/agent : 可一次签好 5 笔 nonce=10,11,12,13,14 全部广播
          ⇒ ★ **同区块内出现多个连续 nonce** 是程序化操作的强信号，人类几乎做不到

    ⚠️ 与 f03_gas 同样存在特征可得性差异: 只有 tx_from 活动才有真实 nonce。

    ============================================================
    二、特征定义 (逻辑 / 数据源 / 业务意义 / 特征生成时间)
    ============================================================
      数据源: 统一活动流的 nonce / blockNumber / kind

      | 字段                  | 计算逻辑                          | 业务意义              | 生成时间 |
      |-----------------------|-----------------------------------|-----------------------|----------|
      | f_nonce_gap_ratio     | 相邻 nonce 差 >1 的占比           | 跳号（有其他并发源）  | 活动落块后 |
      | f_nonce_seq_ratio     | 相邻 nonce 差 ==1 的占比          | 严格顺序发送          | 同上     |
      | f_nonce_burst_max     | 同区块内最多几笔连续 nonce        | ★并发广播强度         | 同上     |
      | f_nonce_burst_ratio   | 同区块多笔 nonce 的区块占比       | ★程序化批量的频率     | 同上     |
      | f_nonce_span          | max(nonce)-min(nonce)             | 观测期内的发送量      | 同上     |

    ============================================================
    三、特征调用方式
    ============================================================
      from src.feature.f04_nonce import FeatureNonce
      feat = FeatureNonce().get_feature(acts)

    ============================================================
    四、使用注意事项
    ============================================================
      1) 只用 kind=='tx_from' 的活动 —— 其余活动的 nonce 字段无意义（填 0）。
      2) 无有效样本时返回 NaN，交由下游 imputer 处理。
      3) f_nonce_burst_max 是本族判别力最强的量，
         但 Base 出块仅 2s，人类连点两笔也可能落同区块 —— 需配合 f01 的 Δt 一起看。

    Update:
'''

import numpy as np
from collections import defaultdict


# ================================================================
#  FeatureNonce — Nonce 并发 (5 特征)
# ================================================================
class FeatureNonce(object):
    """Nonce 并发特征类。回答「这个地址怎么排队发交易」。"""

    def __init__(self):
        """无状态特征类，构造无需参数。"""
        pass

    # ============================================================
    #  get_feature() — 装配全部 Nonce 特征
    # ============================================================
    def get_feature(self, acts: list) -> dict:
        """
        计算该地址的 Nonce 并发特征

        Args:
            acts: 该地址的活动流
        Returns:
            dict: 5 个特征
        """
        NAN = {k: np.nan for k in
               ('f_nonce_gap_ratio', 'f_nonce_seq_ratio', 'f_nonce_burst_max',
                'f_nonce_burst_ratio', 'f_nonce_span')}
        # 1. 只用自己发出的交易
        tx = [a for a in acts if a.get('kind') == 'tx_from']
        if len(tx) < 2:
            return NAN

        nonces = np.array([a.get('nonce', 0) for a in tx], dtype=float)
        blocks = [a['blockNumber'] for a in tx]
        dn = np.diff(nonces)

        # 同区块内的连续 nonce 串长度
        per_blk = defaultdict(list)
        for b, n in zip(blocks, nonces):
            per_blk[b].append(n)
        burst = [len(v) for v in per_blk.values()]

        return {
            'f_nonce_gap_ratio': float((dn > 1).sum() / len(dn)),
            'f_nonce_seq_ratio': float((dn == 1).sum() / len(dn)),
            'f_nonce_burst_max': int(max(burst)) if burst else 0,
            'f_nonce_burst_ratio':
                float(sum(1 for x in burst if x > 1) / max(len(burst), 1)),
            'f_nonce_span': float(nonces.max() - nonces.min()),
        }
