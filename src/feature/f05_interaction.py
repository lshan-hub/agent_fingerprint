#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     : agent_fingerprint
@File        : f05_interaction.py
@Description :
    ============================================================
    一、特征体系 (业务场景分类)
    ============================================================
      类别 5/6:「交互结构」(9 特征) —— 回答: 这个地址「跟谁、怎么打交道」?
      课题依据: 三类主体的交互多样性本质不同
        脚本 bot  : 反复调同几个合约的同几个方法 ⇒ 高重复、低熵
        AI agent  : 按任务调用不同服务 ⇒ 中等多样性
        人类       : 探索性使用，交互对手与方法都散 ⇒ 高多样性

      调研依据: 已有文献把链上特征分为四类 ——
        Interaction / Derived Network / Transfer-based / Temporal，
        本族对应其中的 Interaction 与 Network 部分。

    ============================================================
    二、特征定义 (逻辑 / 数据源 / 业务意义 / 特征生成时间)
    ============================================================
      数据源: 统一活动流的 to / methodId / isError / kind

      | 字段                    | 计算逻辑                       | 业务意义              | 生成时间 |
      |-------------------------|--------------------------------|-----------------------|----------|
      | f_unique_to_ratio       | 不同对手数 / 活动数            | ★交互多样性           | 活动落块后 |
      | f_to_entropy            | 对手分布的归一化熵             | 集中还是分散          | 同上     |
      | f_top_to_share          | 最高频对手的占比               | ★接近1=专用脚本       | 同上     |
      | f_method_entropy        | methodId 分布的归一化熵        | ★方法多样性           | 同上     |
      | f_top_method_share      | 最高频方法的占比               | 单一功能 vs 多功能    | 同上     |
      | f_method_nunique        | 不同方法数                     | 功能广度              | 同上     |
      | f_failed_ratio          | isError=='1' 的占比            | ★抢跑失败率高=MEV bot | 同上     |
      | f_kind_entropy          | 活动类型(tx_from/to/event)熵   | 链上足迹形态复杂度    | 同上     |
      | f_repeat_pair_ratio     | (to,method) 组合的重复率       | ★固定套路 = 脚本      | 同上     |

    ============================================================
    三、特征调用方式
    ============================================================
      from src.feature.f05_interaction import FeatureInteraction
      feat = FeatureInteraction().get_feature(acts)

    ============================================================
    四、使用注意事项
    ============================================================
      1) methodId 来自 SQD 的 sighash（calldata 前 4 字节）。
         event 类活动没有 sighash，本模块填的是事件 topic0 前 10 位 —— 语义不同，
         但对「重复度」这个维度仍然可比（都是「在做同一件事」的代理）。
      2) f_failed_ratio 对 MEV bot 判别力强（抢跑失败会回滚但仍上链），
         但 event 类活动恒为成功 ⇒ 对 agent 组该特征恒为 0，属结构性差异。
      3) f_kind_entropy 是**元特征**（描述活动形态构成），
         判别力极强但反映的是协议形态而非行为，论文需分开讨论。

    Update:
'''

import math

import numpy as np
from collections import Counter


# ================================================================
#  FeatureInteraction — 交互结构 (9 特征)
# ================================================================
class FeatureInteraction(object):
    """交互结构特征类。回答「这个地址跟谁、怎么打交道」。"""

    def __init__(self):
        """无状态特征类，构造无需参数。"""
        pass

    @staticmethod
    def norm_entropy(counts) -> float:
        """归一化 Shannon 熵 ∈[0,1]。1=完全均匀，0=全集中。"""
        c = np.asarray(list(counts), dtype=float)
        total = c.sum()
        if total == 0 or len(c) == 0:
            return np.nan
        p = c[c > 0] / total
        h = -(p * np.log(p)).sum()
        hmax = math.log(len(c))
        return float(h / hmax) if hmax > 0 else 0.0

    # ============================================================
    #  get_feature() — 装配全部交互特征
    # ============================================================
    def get_feature(self, acts: list) -> dict:
        """
        计算该地址的交互结构特征

        Args:
            acts: 该地址的活动流
        Returns:
            dict: 9 个特征
        """
        if not acts:
            return {}
        n = len(acts)
        tos = [a.get('to') or '' for a in acts]
        methods = [a.get('methodId') or '0x' for a in acts]
        kinds = [a.get('kind') or 'unknown' for a in acts]
        pairs = list(zip(tos, methods))

        c_to, c_m, c_k = Counter(tos), Counter(methods), Counter(kinds)
        c_p = Counter(pairs)

        return {
            'f_unique_to_ratio': len(c_to) / n,
            'f_to_entropy': self.norm_entropy(c_to.values()),
            'f_top_to_share': c_to.most_common(1)[0][1] / n,
            'f_method_entropy': self.norm_entropy(c_m.values()),
            'f_top_method_share': c_m.most_common(1)[0][1] / n,
            'f_method_nunique': len(c_m),
            'f_failed_ratio': float(sum(1 for a in acts
                                        if a.get('isError') == '1') / n),
            'f_kind_entropy': self.norm_entropy(c_k.values()),
            # 重复率 = 1 - 唯一组合占比；越高说明越是固定套路
            'f_repeat_pair_ratio': 1.0 - len(c_p) / n,
        }
