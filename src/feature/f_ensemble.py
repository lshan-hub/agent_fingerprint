#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project     : agent_fingerprint
@File        : f_ensemble.py
@Description :
    ============================================================
    一、特征装配总览
    ============================================================
      把 6 个特征族装配成一张宽表（57 特征 + 3 规模列 + 2 标识列）。

      族 | 模块              | 类                   | 特征数 | 回答什么问题
      ---|-------------------|----------------------|--------|------------------------
      1  | f01_latency.py    | FeatureLatency       | 18     | ★最快能多快做下一个决策
      2  | f02_rhythm.py     | FeatureRhythm        | 12     | 什么时候做事
      3  | f03_gas.py        | FeatureGas           | 7      | 怎么出价
      4  | f04_nonce.py      | FeatureNonce         | 5      | 怎么排队发交易
      5  | f05_interaction.py| FeatureInteraction   | 9      | 跟谁、怎么打交道
      6  | f06_value.py      | FeatureValue         | 6      | 花多少钱、怎么取整
      ---|-------------------|----------------------|--------|------------------------
                                                  合计 57 特征

    ============================================================
    二、特征集划分（供建模层直接引用）
    ============================================================
      MODEL_FEATURES        全部 57 个 —— 主模型用
      LATENCY_ONLY_FEATURES 仅族 1 的 18 个 —— 单独验证 H1 的判别力
      NO_META_FEATURES      剔除「元特征」后的集合 —— 见下方警告
      SCALE_FEATURES        n_acts / span_days / acts_per_day，**不进模型**
                            （避免模型用「活动多不多」作弊）

      🔴 元特征警告（论文必须分开讨论）:
        f_txfrom_ratio / f_gas_avail_ratio / f_kind_entropy 这三个描述的是
        **数据可得性本身**，而非行为差异。它们对 agent 组判别力极强，因为
        x402 付款方走元交易（Facilitator 代付）⇒ 结构性没有 gasPrice/nonce。
        · 若目标是「做一个能用的分类器」→ 保留，它们是合法信号
        · 若目标是「论证 agent 有独特行为指纹」→ 必须剔除后复测，
          否则结论会退化成「agent 用了不同的支付协议」这种同义反复
        ⇒ 用 NO_META_FEATURES 跑一遍，报告两者差值。

    ============================================================
    三、特征调用方式
    ============================================================
      from src.feature.f_ensemble import FeatureEnsemble

      Ins = FeatureEnsemble()
      feat = Ins.get_feature(acts)                    # 单地址 → dict
      df   = Ins.build_table(chain='base')            # 全量 → DataFrame
      df   = Ins.run(chain='base')                    # 全量 + 落盘 + 校验

    ============================================================
    四、使用注意事项
    ============================================================
      1) 输入是 src/data_fetch/fetch_activity.py 产出的**统一活动流**，
         不是 fetch_txs.py 的交易史 —— 后者对 Olas/x402 会漏采（实测为 0）。
      2) 活动数 < config.MIN_TX_FOR_FEATURES(20) 的地址直接剔除。
      3) 各族返回 {} 时视为该族全缺失，用 NaN 填充后交由下游 imputer 处理，
         **不要在此处填 0**（会造成假信号）。
      4) 落盘到 data/features/features_{tag}.csv，供 src/modeling/ 读取。

    Update:
'''

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils.common import log  # noqa: E402

from src.feature.f01_latency import FeatureLatency  # noqa: E402
from src.feature.f02_rhythm import FeatureRhythm  # noqa: E402
from src.feature.f03_gas import FeatureGas  # noqa: E402
from src.feature.f04_nonce import FeatureNonce  # noqa: E402
from src.feature.f05_interaction import FeatureInteraction  # noqa: E402
from src.feature.f06_value import FeatureValue  # noqa: E402


# ================================================================
#  FeatureEnsemble — 6 族装配 (57 特征)
# ================================================================
class FeatureEnsemble(object):
    """
    特征装配器：把 6 个族装成一张宽表。

    核心函数 (共 3 个):
      ① get_feature(acts)              单地址 → 57 特征 dict
      ② build_table(chain, synthetic)  批量 → DataFrame
      ③ run(chain, synthetic)          ②+落盘+摘要+校验

    校验函数: validate_table

    Args:
        min_acts: 最低活动数门槛；缺省 config.MIN_TX_FOR_FEATURES
        bands:    (bot 带上界, agent 带上界)；缺省取 config.BAND_*
    """

    # ---- 各族的特征列（与各模块 get_feature 的返回键一一对应）----
    F01_LATENCY = ['f_dt_min', 'f_dt_p01', 'f_dt_p05', 'f_dt_p10', 'f_dt_p25',
                   'f_dt_p50', 'f_dt_p75', 'f_dt_mean', 'f_dt_std', 'f_dt_iqr',
                   'f_dt_cv', 'f_log_dt_median', 'f_frac_lt2s', 'f_frac_2to10s',
                   'f_frac_gt10s', 'f_m2_p05', 'f_m2_p50', 'f_m2_n']
    F02_RHYTHM = ['f_burstiness', 'f_memory_coef', 'f_same_block_ratio',
                  'f_consec_block_ratio', 'f_longest_silence_h', 'f_entropy_24h',
                  'f_entropy_dow', 'f_night_ratio', 'f_peak_hour_share',
                  'f_active_hours', 'f_active_days', 'f_span_days']
    F03_GAS = ['f_gas_price_cv', 'f_gas_price_mode_share', 'f_gas_price_median',
               'f_gas_price_p95_p50', 'f_gas_nunique_ratio', 'f_gas_avail_ratio',
               'f_txfrom_ratio']
    F04_NONCE = ['f_nonce_gap_ratio', 'f_nonce_seq_ratio', 'f_nonce_burst_max',
                 'f_nonce_burst_ratio', 'f_nonce_span']
    F05_INTERACTION = ['f_unique_to_ratio', 'f_to_entropy', 'f_top_to_share',
                       'f_method_entropy', 'f_top_method_share',
                       'f_method_nunique', 'f_failed_ratio', 'f_kind_entropy',
                       'f_repeat_pair_ratio']
    F06_VALUE = ['f_value_zero_ratio', 'f_round_value_ratio',
                 'f_value_median_eth', 'f_value_cv', 'f_value_nunique_ratio',
                 'f_micro_value_ratio']

    MODEL_FEATURES = (F01_LATENCY + F02_RHYTHM + F03_GAS
                      + F04_NONCE + F05_INTERACTION + F06_VALUE)
    LATENCY_ONLY_FEATURES = F01_LATENCY

    # 🔴 元特征：描述「数据可得性」而非行为，论文需分开讨论（见模块文档 二）
    META_FEATURES = ['f_txfrom_ratio', 'f_gas_avail_ratio', 'f_kind_entropy']
    # ⚠️ 类体内的推导式看不到其他类变量（Python 作用域规则），
    #    所以这里用显式循环而非 [f for f in MODEL_FEATURES if ...]
    NO_META_FEATURES = list(MODEL_FEATURES)
    for _m in META_FEATURES:
        if _m in NO_META_FEATURES:
            NO_META_FEATURES.remove(_m)
    del _m

    # 🔴 窗口依赖特征：各组采集窗口不等长（human ×12，见 config.GROUP_WINDOW_MULT）。
    #    f02 内部已把它们除以本组窗口做归一化，但归一化不等于完全无泄漏 ——
    #    若模型在剔除它们后指标明显下降，说明判别力有一部分来自采集设计而非行为。
    # 🔴 2026-09-04 扩充：前 3 个是「绝对跨度」，后 3 个是「周期覆盖度」——
    #    后者更隐蔽但危害更大。entropy_dow 需要窗口 >=7 天才有意义:
    #    6.6 小时窗口跨不过一天 ⇒ dow 直方图只有 1 个非零桶 ⇒ 熵恒为 0；
    #    5.5 天窗口 ⇒ 熵天然为正。模型只要学「dow 熵 > 0 ⇒ human」就能作弊。
    #    实测该特征一度是 behavior 集第一重要特征（gain 31.5%），纯属窗口伪影。
    #    entropy_24h / active_hours 同理，需要窗口 >=24 小时。
    #    ⇒ 对策见 config.MIN_WINDOW_FOR_CYCLE_BLOCKS: 三组窗口统一到 >=7 天，
    #      让这些特征重新可比；在此之前它们必须留在本清单里被消融掉。
    WINDOW_DEP_FEATURES = ['f_span_days', 'f_active_days', 'f_longest_silence_h',
                           'f_entropy_dow', 'f_entropy_24h', 'f_active_hours']
    NO_WINDOW_FEATURES = list(MODEL_FEATURES)
    for _w in WINDOW_DEP_FEATURES:
        if _w in NO_WINDOW_FEATURES:
            NO_WINDOW_FEATURES.remove(_w)
    del _w

    # 最严格对照：元特征 + 窗口依赖特征全剔除，只剩纯行为特征
    CLEAN_FEATURES = list(MODEL_FEATURES)
    for _c in META_FEATURES + WINDOW_DEP_FEATURES:
        if _c in CLEAN_FEATURES:
            CLEAN_FEATURES.remove(_c)
    del _c

    # ============================================================
    # 🔴🔴 BEHAVIOR_FEATURES —— 论文主结论应引用的特征集
    # ============================================================
    # 【为什么还需要比 clean 更严的一档】
    #   剔除显式元特征还不够。XGBoost 原生把「缺失」当作一个分裂方向，
    #   所以缺失率按组悬殊的特征，即使数值本身无判别力，
    #   光靠「有没有值」就能标识组别。
    #   实测 base 链缺失率（2026-09-13 全量 30,000 样本复核；
    #   括号内是只算「缺失敏感成员」的口径，即图 4.3 与论文 4.5 节采用的口径，
    #   剔除 f_gas_avail_ratio / f_txfrom_ratio / f_value_zero_ratio 三个恒可算项）:
    #       f03_gas   : agent 70.3% (98.5%) / bot 0.0% / human 0.0%
    #       f04_nonce : agent 100%  (100%)  / bot 0.04% / human 0.0%
    #       f06_value : agent 83.3% (100%)  / bot 33.7% (40.4%) / human 35.9% (43.1%)
    #   ⇒ clean 集 48 个特征里仍有 15 个缺失率组间差 >30pp
    #     （behavior 相对 clean 共剔 16 维，其中 f_value_zero_ratio 恒可算）。
    #   🔴 引用时必须注明口径 —— 两个分母最大相差约 28 个百分点。
    #     模型学到的「nonce 缺失 ⇒ agent」，语义上等同于已被剔除的
    #     f_txfrom_ratio（agent 走 meta-transaction 不自己发交易），
    #     换了个形式重新泄漏进来。
    #
    # 【构成】 只保留三组缺失率都 <5% 的族: f01 延迟 + f02 节律 + f05 交互，
    #         再去掉 3 个窗口依赖特征。这些特征三组都实际算得出来，
    #         判别力只能来自行为本身。
    #
    # 【代价】 放弃了 gas/nonce/value 三族的真实信号 —— 对 bot 和 human
    #         这些族是有效的，这里是为了可比性主动牺牲。
    #         论文应同时报告 clean（含全部信号，上界）与 behavior（无泄漏，下界）。
    BEHAVIOR_FEATURES = list(F01_LATENCY + F02_RHYTHM + F05_INTERACTION)
    for _b in META_FEATURES + WINDOW_DEP_FEATURES:
        if _b in BEHAVIOR_FEATURES:
            BEHAVIOR_FEATURES.remove(_b)
    del _b

    # ============================================================
    # 🔴🔴🔴 TIMING_FEATURES —— 最保守的一档，论文的"下界"
    # ============================================================
    # 【为什么 behavior 集还不够保守】
    #   behavior 集含 f05_interaction，其中 f_to_entropy 的实测中位数:
    #       agent 0.000 / bot 0.408 / human 0.791
    #   agent 恒为 0 不是因为「agent 真的只跟一个对象打交道」，
    #   而是因为该组 98% 的活动是 USDC 合约的 AuthorizationUsed 事件 ——
    #   交互对象被采集方式钉死成了 USDC 合约地址。
    #   这是「协议形态泄漏」的第三种形式: 不是元特征、也不是缺失模式，
    #   而是一个算得出来、但取值由采集口径决定的真实数值。
    #
    # 【构成】 只保留纯时间行为: f01 延迟 + f02 节律，去掉窗口依赖项。
    #         「什么时候动、隔多久动一次」不依赖于我们扫了哪种活动类型。
    #
    # 【如何解读】 这是本课题结论的**下界**。若此集仍显著优于随机
    #            （三分类随机 macro-F1 ≈ 0.33），则「时间行为本身可区分三类主体」
    #            这一最弱形式的论点成立，且无法用任何采集口径差异解释。
    TIMING_FEATURES = list(F01_LATENCY + F02_RHYTHM)
    for _t in META_FEATURES + WINDOW_DEP_FEATURES:
        if _t in TIMING_FEATURES:
            TIMING_FEATURES.remove(_t)
    del _t

    # 规模类：不进模型，避免用「活动多不多」作弊
    SCALE_FEATURES = ['n_acts', 'span_days_raw', 'acts_per_day']

    FAMILIES = {
        'f01_latency': F01_LATENCY, 'f02_rhythm': F02_RHYTHM,
        'f03_gas': F03_GAS, 'f04_nonce': F04_NONCE,
        'f05_interaction': F05_INTERACTION, 'f06_value': F06_VALUE,
    }

    def __init__(self, min_acts: int = None, bands: tuple = None,
                 chain: str = None):
        """
        构造: 实例化 6 个族的特征类。

        Args:
            min_acts: 最低活动数门槛
            bands:    (bot 带上界, agent 带上界)
            chain:    链名，决定 f02 归一化用的出块秒数；缺省 config.PRIMARY_CHAIN
        Returns:
            None
        """
        self.min_acts = min_acts or config.MIN_TX_FOR_FEATURES
        self.bands = bands or (config.BAND_BOT[1], config.BAND_AGENT[1])
        self.chain = chain or config.PRIMARY_CHAIN
        bt = config.SQD_BLOCK_TIME_MEASURED.get(self.chain, 2.0)

        self.f01 = FeatureLatency(bands=self.bands)
        self.f02 = FeatureRhythm(block_time=bt)
        self.f03 = FeatureGas()
        self.f04 = FeatureNonce()
        self.f05 = FeatureInteraction()
        self.f06 = FeatureValue()

        self.n_skipped = 0

    # ============================================================
    #  ① get_feature() — 单地址装配
    # ============================================================
    def get_feature(self, acts: list) -> dict:
        """
        ① 把一个地址的活动流装配成 57 个特征

        逐族调用 → 冲突守卫（键重名 fail-fast）→ 缺失族补 NaN

        Args:
            acts: 该地址的活动流 list[dict]
        Returns:
            dict: 57 特征 + 3 规模列；活动数不足时返回 {}
        """
        if len(acts) < self.min_acts:
            self.n_skipped += 1
            return {}

        # 统一按 (timeStamp, txIndex) 升序 —— 各族都依赖这个前提
        acts = sorted(acts, key=lambda a: (a['timeStamp'], a.get('txIndex', 0)))

        ts = np.array([a['timeStamp'] for a in acts], dtype=float)
        span_s = float(ts[-1] - ts[0])
        out = {
            'n_acts': len(acts),
            'span_days_raw': span_s / 86400,
            'acts_per_day': len(acts) / max(span_s / 86400, 1e-9),
            # 采集窗口长度（块）—— 不是特征，供 validate 做窗口充分性体检:
            # 各组窗口若不等长或不足一周，周期类特征会退化成窗口伪影
            'win_blocks': int(acts[0].get('win_hi', 0)
                              - acts[0].get('win_lo', 0)),
        }

        # 逐族计算，键冲突 fail-fast（对齐参考框架的冲突守卫）
        for name, ins in (('f01_latency', self.f01), ('f02_rhythm', self.f02),
                          ('f03_gas', self.f03), ('f04_nonce', self.f04),
                          ('f05_interaction', self.f05), ('f06_value', self.f06)):
            block = ins.get_feature(acts) or {}
            dup = set(block) & set(out)
            if dup:
                raise ValueError(f'特征键冲突 {name}: {sorted(dup)}')
            out.update(block)

        # f01 返回空 ⇒ Δt 样本不足，该地址不可用
        if not any(k in out for k in self.F01_LATENCY):
            self.n_skipped += 1
            return {}

        # 缺失族补 NaN，保证列对齐
        for f in self.MODEL_FEATURES:
            out.setdefault(f, np.nan)
        return out

    # ============================================================
    #  ② build_table() — 批量装配
    # ============================================================
    def build_table(self, chain: str = None,
                    synthetic: bool = False) -> pd.DataFrame:
        """
        ② 遍历三组地址的活动流，构建特征宽表

        Args:
            chain / synthetic
        Returns:
            pd.DataFrame: [address, group, ...57 特征, ...3 规模列]
        """
        base = config.DATA_RAW / ('synthetic' if synthetic
                                  else (chain or config.PRIMARY_CHAIN))
        if not base.exists():
            log(f'找不到活动流目录 {base}', 'err')
            log('先跑：python3 src/data_fetch/fetch_activity.py', 'info')
            sys.exit(1)

        self.n_skipped = 0
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
                    acts = json.loads(fp.read_text())
                except json.JSONDecodeError:
                    continue
                feat = self.get_feature(acts)
                if not feat:
                    continue
                feat['address'] = fp.stem
                feat['group'] = group
                rows.append(feat)
                n_ok += 1
            log(f'[{group}] {n_ok}/{len(files)} 个地址通过 ≥{self.min_acts} '
                f'条活动的门槛', 'info')

        if not rows:
            log('没有地址通过门槛。检查 MIN_TX_FOR_FEATURES 或活动流数据。', 'err')
            sys.exit(1)

        df = pd.DataFrame(rows)
        cols = ['address', 'group'] + self.SCALE_FEATURES + self.MODEL_FEATURES
        return df[[c for c in cols if c in df.columns]]

    # ============================================================
    #  ③ run() — 装配 + 落盘 + 摘要 + 校验
    # ============================================================
    def save(self, df: pd.DataFrame, tag: str) -> Path:
        """写入 data/features/features_{tag}.csv。"""
        out = config.DATA_FEAT / f'features_{tag}.csv'
        df.to_csv(out, index=False)
        log(f'写入 {out.relative_to(config.ROOT)}  '
            f'({len(df)} 行 × {len(df.columns)} 列)', 'ok')
        return out

    def summarize(self, df: pd.DataFrame) -> None:
        """打印分组样本量 + 各族关键指标的分组中位数。"""
        print()
        print('各组样本量：')
        print(df['group'].value_counts().to_string())
        print()
        print('★ 各族关键指标的分组中位数：')
        key = ['f_dt_p05', 'f_frac_2to10s', 'f_burstiness', 'f_entropy_24h',
               'f_txfrom_ratio', 'f_top_method_share', 'f_value_zero_ratio']
        key = [k for k in key if k in df.columns]
        print(df.groupby('group')[key].median().round(3).to_string())
        print()

    def run(self, chain: str = None, synthetic: bool = False) -> pd.DataFrame:
        """
        ③ 完整流程：装配 → 落盘 → 摘要 → 校验

        Args:
            chain / synthetic
        Returns:
            pd.DataFrame
        """
        tag = 'synthetic' if synthetic else (chain or config.PRIMARY_CHAIN)
        log(f'装配特征：{tag}（6 族 / {len(self.MODEL_FEATURES)} 特征）', 'step')
        df = self.build_table(chain, synthetic)
        self.save(df, tag)
        self.summarize(df)
        self.validate_table(df)
        log('下一步：python3 src/modeling/train_xgb.py'
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
                       context: str = '特征宽表') -> bool:
        """
        校验特征宽表

        检查项:
          1. 非空 / 57 个建模特征列齐全 / 地址无重复
          2. 比例类特征落在 [0,1]；burstiness / memory_coef 落在 [-1,1]
          3. ⚠️ 各族的全缺失情况（按组统计，暴露结构性缺失）
          4. ⚠️ 分组样本量与类别平衡
          5. 🔴 元特征警告

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
            issues.append(f'缺建模特征列（{len(miss)} 个）: {miss[:6]}')
            return self._emit(context, issues, warns)
        if int(df['address'].duplicated().sum()):
            issues.append('存在重复地址')

        # 2. 取值范围
        ratio_cols = [c for c in self.MODEL_FEATURES
                      if c.endswith('_ratio') or c.endswith('_share')
                      or c.startswith('f_frac_') or c.startswith('f_entropy')
                      or c == 'f_to_entropy' or c == 'f_method_entropy'
                      or c == 'f_kind_entropy']
        for c in ratio_cols:
            s = df[c].dropna()
            if len(s) and (s.min() < -1e-9 or s.max() > 1 + 1e-9):
                issues.append(f'{c} 超出 [0,1]：[{s.min():.3f}, {s.max():.3f}]')
        for c in ('f_burstiness', 'f_memory_coef'):
            s = df[c].dropna()
            if len(s) and (s.min() < -1 - 1e-9 or s.max() > 1 + 1e-9):
                issues.append(f'{c} 超出 [-1,1]：[{s.min():.3f}, {s.max():.3f}]')

        # 3. 各族按组的缺失率
        for fam, cols in self.FAMILIES.items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            by_g = df.groupby('group')[cols].apply(
                lambda x: float(x.isna().all(axis=1).mean()))
            bad = {g: f'{v:.0%}' for g, v in by_g.items() if v > 0.5}
            if bad:
                warns.append(f'{fam} 整族缺失: {bad} —— 结构性缺失，'
                             f'论文需交代「特征可得性差异」')

        # 4. 分组
        vc = df['group'].value_counts()
        lost = set(config.GROUPS) - set(vc.index)
        if lost:
            warns.append(f'缺少分组 {sorted(lost)} —— 三分类需三组齐全')
        for g, n in vc.items():
            if n < 30:
                warns.append(f'{g} 组仅 {n} 个样本，交叉验证会不稳定')
        if len(vc) > 1 and vc.max() / vc.min() > 5:
            warns.append(f'类别不平衡 {dict(vc)} —— 需 class_weight + macro-F1')

        if self.n_skipped:
            warns.append(f'{self.n_skipped} 个地址因活动数不足被跳过')
        warns.append(f'🔴 元特征 {self.META_FEATURES} 描述数据可得性而非行为，'
                     f'建议用 NO_META_FEATURES 复测并报告差值')

        # 🔴 窗口充分性体检 —— 周期类特征需要窗口覆盖完整周期才有意义
        if 'win_blocks' in df.columns and 'group' in df.columns:
            wb = df.groupby('group')['win_blocks'].median()
            need = config.MIN_WINDOW_FOR_CYCLE_BLOCKS
            short = {g: int(v) for g, v in wb.items() if v < need}
            if short:
                issues.append(
                    f'🔴 窗口不足一周的组: {short}（需 >= {need:,} 块）。'
                    f'entropy_dow/entropy_24h/active_hours 会退化成窗口伪影 —— '
                    f'6.6h 窗口跨不过一天，dow 熵恒为 0')
            if len(wb) > 1 and wb.max() / max(wb.min(), 1) > 1.5:
                issues.append(
                    f'🔴 各组窗口长度不等（{dict(wb.astype(int))}），'
                    f'周期类特征不可比；请统一 config.GROUP_WINDOW_MULT')

        # 🔴 缺失模式泄漏体检 —— XGBoost 把「缺失」当分裂方向，
        #    缺失率组间悬殊的特征光靠「有没有值」就能标识组别
        if 'group' in df.columns:
            leak = []
            for c in self.CLEAN_FEATURES:
                if c not in df.columns:
                    continue
                m = df.groupby('group')[c].apply(lambda s: s.isna().mean())
                if len(m) > 1 and (m.max() - m.min()) > 0.30:
                    leak.append(c)
            if leak:
                warns.append(
                    f'🔴 clean 集中 {len(leak)}/{len(self.CLEAN_FEATURES)} 个特征'
                    f'缺失率组间差 >30pp，「有没有值」本身即可标识组别；'
                    f'论文主结论请引用 BEHAVIOR_FEATURES'
                    f'（{len(self.BEHAVIOR_FEATURES)} 个，三组均可算）')
        return self._emit(context, issues, warns)


# ---- 向后兼容：建模层直接 import 这几个常量 ----
MODEL_FEATURES = FeatureEnsemble.MODEL_FEATURES
LATENCY_ONLY_FEATURES = FeatureEnsemble.LATENCY_ONLY_FEATURES
NO_META_FEATURES = FeatureEnsemble.NO_META_FEATURES
NO_WINDOW_FEATURES = FeatureEnsemble.NO_WINDOW_FEATURES
CLEAN_FEATURES = FeatureEnsemble.CLEAN_FEATURES
BEHAVIOR_FEATURES = FeatureEnsemble.BEHAVIOR_FEATURES
TIMING_FEATURES = FeatureEnsemble.TIMING_FEATURES


def main():
    ap = argparse.ArgumentParser(description='特征装配（6 族 / 57 特征）')
    ap.add_argument('--chain', default=config.PRIMARY_CHAIN,
                    choices=list(config.CHAINS))
    ap.add_argument('--synthetic', action='store_true', help='使用合成数据')
    ap.add_argument('--min-acts', type=int, default=None, help='最低活动数门槛')
    args = ap.parse_args()
    FeatureEnsemble(min_acts=args.min_acts, chain=args.chain).run(chain=args.chain,
                                                synthetic=args.synthetic)


if __name__ == '__main__':
    main()
