#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     : agent_fingerprint
@File        : f06_value.py
@Description :
    ============================================================
    一、特征体系 (业务场景分类)
    ============================================================
      类别 6/6:「金额偏好」(6 特征) —— 回答: 这个地址「花多少钱、怎么取整」?
      课题依据:
        · 脚本倾向使用**整数金额**（0.01 / 0.1 ETH 的整数倍）——
          人类手动输入也会取整，但脚本的取整**高度一致**
        · x402 微支付有极强的金额签名: 官方口径均单 $0.31，
          实测样本均单 $0.002~0.006 ⇒ ★ 小额高频是 agent 的强特征
        · MEV bot 的 value 多为 0（套利在 calldata 里完成，不转原生币）

    ============================================================
    二、特征定义 (逻辑 / 数据源 / 业务意义 / 特征生成时间)
    ============================================================
      数据源: 统一活动流的 value（wei）

      | 字段                  | 计算逻辑                        | 业务意义               | 生成时间 |
      |-----------------------|---------------------------------|------------------------|----------|
      | f_value_zero_ratio    | value==0 的占比                 | ★合约调用为主(MEV/agent)| 活动落块后 |
      | f_round_value_ratio   | 非零 value 是 1e16 整数倍占比   | ★整数偏好 = 脚本       | 同上     |
      | f_value_median_eth    | 非零 value 中位数（ETH）        | 典型金额档位           | 同上     |
      | f_value_cv            | 非零 value 的变异系数           | 金额稳定性             | 同上     |
      | f_value_nunique_ratio | 不同金额数 / 非零笔数           | 金额多样性             | 同上     |
      | f_micro_value_ratio   | 非零 value < 1e15 wei 的占比    | ★微支付倾向            | 同上     |

    ============================================================
    三、特征调用方式
    ============================================================
      from src.feature.f06_value import FeatureValue
      feat = FeatureValue().get_feature(acts)

    ============================================================
    四、使用注意事项
    ============================================================
      1) value 是**原生币**（Base 上是 ETH）的转账额，单位 wei。
         x402 的 USDC 支付**不体现在 value 里**（是 ERC-20 转账，在事件里），
         所以 x402 付款方的 f_value_zero_ratio 会接近 1 —— 这是正常的。
      2) 「整数偏好」阈值取 1e16 wei = 0.01 ETH。更细的档位（0.001）可按需调整。
      3) 全零 value 时各分布量返回 NaN，交由下游 imputer 处理。

    Update:
'''

import numpy as np


# ================================================================
#  FeatureValue — 金额偏好 (6 特征)
# ================================================================
class FeatureValue(object):
    """
    金额偏好特征类。

    Args:
        round_unit: 「整数偏好」的判定单位（wei），缺省 1e16 = 0.01 ETH
        micro_unit: 「微支付」的判定上限（wei），缺省 1e15 = 0.001 ETH
    """

    ETH = 1e18

    def __init__(self, round_unit: float = 1e16, micro_unit: float = 1e15):
        """构造: 存整数偏好与微支付的判定阈值。"""
        self.round_unit = round_unit
        self.micro_unit = micro_unit

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
    #  get_feature() — 装配全部金额特征
    # ============================================================
    def get_feature(self, acts: list) -> dict:
        """
        计算该地址的金额偏好特征

        Args:
            acts: 该地址的活动流
        Returns:
            dict: 6 个特征
        """
        if not acts:
            return {}
        vals = np.array([self._f(a.get('value')) for a in acts])
        nz = vals[vals > 0]
        out = {'f_value_zero_ratio': float((vals == 0).sum() / len(vals))}

        if len(nz) == 0:
            out.update({k: np.nan for k in
                        ('f_round_value_ratio', 'f_value_median_eth',
                         'f_value_cv', 'f_value_nunique_ratio',
                         'f_micro_value_ratio')})
            return out

        mean = nz.mean()
        out.update({
            'f_round_value_ratio': float((nz % self.round_unit == 0).sum() / len(nz)),
            'f_value_median_eth': float(np.median(nz)) / self.ETH,
            'f_value_cv': float(nz.std() / mean) if mean > 0 else np.nan,
            'f_value_nunique_ratio': float(len(set(nz.tolist())) / len(nz)),
            'f_micro_value_ratio': float((nz < self.micro_unit).sum() / len(nz)),
        })
        return out
