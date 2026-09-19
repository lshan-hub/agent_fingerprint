#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：robustness_checks.py
@Description:
    稳健性检验 —— 补上 2026-09-13 审计发现的三个论证缺口。

    ============================================================
    为什么需要这三组实验
    ============================================================
    七组特征集消融（train_xgb.py）给出的是「逐级收紧后还剩多少」，
    但它回答不了下面三个问题，而这三个问题恰恰是审稿人最容易打穿的地方：

    【缺口 A：消融 Δ 在指标饱和时不可解读】
      论文原写法：「剔除元特征族后 Δ=+0.0002，说明协议形态捷径贡献近乎为零」。
      这个推断在「多条捷径互为冗余 + model 集已达 0.9997」时无效 ——
      拿掉一条，剩下的立刻补位，Δ 自然≈0。
      ⇒ 实验 A：让每条捷径**单独成模**，看它自己能到多少。
        实测 16 维缺失敏感特征单独就有 macro-F1 0.989、agent 召回 0.9998，
        单个 f_gas_avail_ratio 的 agent 召回就有 0.9999。
        结论必须改写成「高度冗余」，而不是「无贡献」。

    【缺口 B：三组 Δt 的观测支撑不等长】
      bot 组按 36 段 × 10,000 块采集 ⇒ 段内 Δt 物理上限约 20,000 秒；
      agent/human 全窗口连续 ⇒ Δt 上限是整个 187 天窗口。
      于是 f_dt_p75 / f_dt_mean / f_dt_std / f_dt_iqr / f_longest_silence_h
      的尾部对 bot 被结构性截断 —— 这是采集设计，不是行为差异。
      ⇒ 实验 B：把三组 Δt 一律 winsorize 到 bot 的段长上限后重训，
        看主结论掉多少。（winsorize 而非丢弃，样本量保持 30,000 不变，
        避免「丢掉最慢的地址」引入二次选择偏差。）

    【缺口 C：七组特征集并不互相独立】
      model / latency / nometa / nowin / clean / behavior / timing
      **七组全部包含完整的 18 维延迟族**，消融只动延迟族以外的部分。
      所以「七组无一例外漏检 56%~68%」不是七个独立证据，
      而是同一个共享内核被重复测了七次。
      ⇒ 实验 C：构造一个**剔除整个延迟族**的对照集
        （behavior − f01 = 节律 6 + 交互 8 = 14 维），
        这才是真正检验「与特征口径无关」的那一组。

    ============================================================
    产出
    ============================================================
      figures/ROBUSTNESS_{链}.md          论文可直接引用的三张表
      figures/ROBUSTNESS_STATS_{链}.json  机读实测值（供对账）

    ============================================================
    口径约定（与 train_xgb.py 严格一致，保证数字可比）
    ============================================================
      · 同一份 features_{链}.csv、同一个 XgbClfModel 管线（Phase 0~5）
      · 同一套种子 config.RANDOM_SEED；分层 5 折 OOF
      · 二分类对照同样走 Phase 5，与表 5.5 同口径

    调用:
      python3 src/verify/robustness_checks.py --chain base
'''

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conf import config  # noqa: E402
from src.utils.common import log  # noqa: E402
from src.feature.f01_latency import FeatureLatency  # noqa: E402
from src.feature.f02_rhythm import FeatureRhythm  # noqa: E402
from src.feature.f_ensemble import FeatureEnsemble as FE  # noqa: E402
from src.modeling.XgbClf import XgbClfModel  # noqa: E402


class RobustnessChecks(object):
    """
    三组稳健性对照实验。

    核心函数 (共 5 个):
      ① exp_a_shortcut_alone()  捷径单独成模 —— 破「Δ≈0 ⇒ 无贡献」
      ② exp_b_dt_censor()       Δt 截断对照 —— 量化采集不对称的影响
      ③ exp_c_no_latency()      无延迟族对照 —— 破「七组独立」
      ④ report()                汇总成 Markdown
      ⑤ run()                   ①~④ + 落盘

    Args:
        chain: 链名，缺省 config.PRIMARY_CHAIN
    """

    # bot 组段长（块）× 出块秒数 = 该组段内 Δt 的物理上限
    @property
    def dt_cap_s(self) -> float:
        bt = config.SQD_BLOCK_TIME_MEASURED.get(self.chain, 2.0)
        return float(config.BOT_COLLECT_SEG_BLOCKS * bt)

    def __init__(self, chain: str = None):
        self.chain = chain or config.PRIMARY_CHAIN
        self.df = pd.DataFrame()
        self.out = {}
        self._tmp = ROOT / 'data' / 'model' / '_robustness_tmp'

    # ============================================================
    #  载入
    # ============================================================
    def load(self) -> pd.DataFrame:
        p = config.DATA_FEAT / f'features_{self.chain}.csv'
        if not p.exists():
            log(f'找不到 {p}；先跑 python3 src/feature/f_ensemble.py', 'err')
            sys.exit(1)
        self.df = pd.read_csv(p)
        log(f'载入 {len(self.df):,} 行 × {len(self.df.columns)} 列', 'ok')
        return self.df

    # ============================================================
    #  统一的训练包装 —— 与 train_xgb 同口径
    # ============================================================
    def _fit(self, name: str, sub: pd.DataFrame, feats: list) -> dict:
        """训练一套特征集并返回精简指标。save_path 用临时目录，不污染 data/model。"""
        use = [f for f in feats if f in sub.columns]
        if not use:
            return {}
        m = XgbClfModel(df_train=sub[['address', 'group'] + use],
                        save_path=self._tmp / name,
                        top_n=min(40, len(use)), min_keep=1, verbose=False)
        try:
            r = m.train()
        except Exception as e:                      # noqa: BLE001
            log(f'[{name}] 训练失败: {type(e).__name__}: {e}', 'err')
            return {}
        bc = r.get('binary_control') or {}
        return {
            'n_nominal': len(use),
            'n_used': r['n_features'],
            'n_samples': r['n_samples'],
            'macro_f1': round(r['oof_macro_f1'], 4),
            'recall': {k: round(v, 4) for k, v in r['oof_recall'].items()},
            'as_human_pct': (None if not bc
                             else round(bc['as_human_pct'], 1)),
            'as_bot_pct': (None if not bc else round(bc['as_bot_pct'], 1)),
        }

    # ============================================================
    #  ① 实验 A —— 捷径单独成模
    # ============================================================
    def exp_a_shortcut_alone(self) -> dict:
        """
        ① 每条捷径单独喂进模型，看它自己能到多少。

        这是对「消融 Δ≈0」的正面反驳：若单独成模就接近满分，
        那么 Δ≈0 反映的是冗余，而不是该捷径没有判别力。

        Returns:
            dict: {对照名: 指标}
        """
        log('=' * 62, 'info')
        log('实验 A — 捷径单独成模（破「Δ≈0 ⇒ 无贡献」）', 'step')
        log('=' * 62, 'info')

        miss_sens = [c for c in FE.CLEAN_FEATURES
                     if c not in FE.BEHAVIOR_FEATURES]
        cases = [
            ('meta3', '仅 3 维协议形态元特征', FE.META_FEATURES),
            ('win6', '仅 6 维窗口依赖特征', FE.WINDOW_DEP_FEATURES),
            ('miss16', '仅 16 维缺失模式特征（gas/nonce/value）', miss_sens),
            ('gas_avail1', '仅 f_gas_avail_ratio 单个特征',
             ['f_gas_avail_ratio']),
            ('silence1', '仅 f_longest_silence_h 单个特征',
             ['f_longest_silence_h']),
        ]
        res = {}
        for key, desc, feats in cases:
            r = self._fit(f'a_{key}', self.df, feats)
            if not r:
                continue
            r['desc'] = desc
            res[key] = r
            log(f'  {desc}: macro-F1={r["macro_f1"]:.4f} '
                f'agent召回={r["recall"].get("agent", float("nan")):.4f}', 'info')
        return res

    # ============================================================
    #  ② 实验 B —— Δt 截断对照
    # ============================================================
    def _recompute_dt(self, cap: float) -> pd.DataFrame:
        """按 cap 对段内 Δt 做 winsorize 后重算 f01 与 f02 中依赖 Δt 的特征。

        只重算「以 Δt 为输入」的列；小时/星期直方图类特征不受 cap 影响，
        直接沿用原表，避免引入无关差异。
        """
        f1 = FeatureLatency(bands=(config.BAND_BOT[1], config.BAND_AGENT[1]))
        f2 = FeatureRhythm(
            block_time=config.SQD_BLOCK_TIME_MEASURED.get(self.chain, 2.0))
        base = config.DATA_RAW / self.chain
        rows = []
        for g in config.GROUPS:
            d = base / g
            if not d.exists():
                continue
            for fp in sorted(d.glob('*.json')):
                try:
                    acts = json.loads(fp.read_text())
                except json.JSONDecodeError:
                    continue
                acts = sorted(acts,
                              key=lambda a: (a['timeStamp'], a.get('txIndex', 0)))
                ts = np.array([a['timeStamp'] for a in acts], dtype=float)
                seg = np.array([a.get('seg', 0) for a in acts], dtype=float)
                blocks = np.array([a['blockNumber'] for a in acts], dtype=float)
                if len(ts) < 2:
                    continue
                d_ = np.diff(ts)
                same = np.diff(seg) == 0
                dt = d_[same & (d_ >= 0)]
                if len(dt) < FeatureLatency.MIN_DT_SAMPLES:
                    continue
                dt = np.minimum(dt, cap)                 # ★ winsorize
                m2 = np.minimum(f1.same_target_deltas(acts), cap)
                pos = dt[dt > 0]
                lt2, mid, gt10 = f1.band_fractions(dt)
                p25 = float(np.percentile(dt, 25))
                p75 = float(np.percentile(dt, 75))
                mean, std = float(dt.mean()), float(dt.std())
                db = np.diff(blocks)
                rows.append({
                    'address': fp.stem, 'group': g,
                    'f_dt_min': float(dt.min()),
                    'f_dt_p01': float(np.percentile(dt, 1)),
                    'f_dt_p05': float(np.percentile(dt, 5)),
                    'f_dt_p10': float(np.percentile(dt, 10)),
                    'f_dt_p25': p25,
                    'f_dt_p50': float(np.percentile(dt, 50)),
                    'f_dt_p75': p75,
                    'f_dt_mean': mean, 'f_dt_std': std, 'f_dt_iqr': p75 - p25,
                    'f_dt_cv': std / mean if mean > 0 else np.nan,
                    'f_log_dt_median': (float(np.median(np.log10(pos)))
                                        if len(pos) else np.nan),
                    'f_frac_lt2s': lt2, 'f_frac_2to10s': mid,
                    'f_frac_gt10s': gt10,
                    'f_m2_p05': (float(np.percentile(m2, 5))
                                 if len(m2) >= 5 else np.nan),
                    'f_m2_p50': (float(np.percentile(m2, 50))
                                 if len(m2) >= 5 else np.nan),
                    'f_m2_n': len(m2),
                    'f_burstiness': f2.burstiness(dt),
                    'f_memory_coef': f2.memory_coefficient(dt),
                    'f_same_block_ratio': float((dt == 0).sum() / len(dt)),
                    'f_consec_block_ratio': float(
                        (db[same] == 1).sum() / max(same.sum(), 1)),
                })
        return pd.DataFrame(rows)

    def exp_b_dt_censor(self) -> dict:
        """
        ② 把三组 Δt 一律 winsorize 到 bot 的段长上限后重训 behavior 集。

        Returns:
            dict: {'baseline': …, 'censored': …, 'cap_s': …}
        """
        log('=' * 62, 'info')
        log(f'实验 B — Δt 截断对照（cap = {self.dt_cap_s:,.0f} 秒）', 'step')
        log('=' * 62, 'info')
        log('  重算段内 Δt 派生特征（需遍历 data/raw，约 30 秒）…', 'info')
        cap_df = self._recompute_dt(self.dt_cap_s)
        if cap_df.empty:
            log('  raw 目录为空，跳过实验 B', 'warn')
            return {}

        beh = FE.BEHAVIOR_FEATURES
        rec = [c for c in beh if c in cap_df.columns]
        oth = [c for c in beh if c not in cap_df.columns]
        merged = cap_df[['address', 'group'] + rec].merge(
            self.df[['address'] + oth], on='address', how='inner')
        log(f'  截断后样本 {len(merged):,}（应与基线一致 = winsorize 不丢地址）',
            'info')

        base = self._fit('b_baseline', self.df, beh)
        cens = self._fit('b_censored', merged, beh)
        for tag, r in (('基线', base), ('Δt 截断', cens)):
            if r:
                log(f'  {tag}: macro-F1={r["macro_f1"]:.4f} '
                    f'agent→human={r["as_human_pct"]}%', 'info')
        return {'cap_s': self.dt_cap_s, 'baseline': base, 'censored': cens}

    # ============================================================
    #  ③ 实验 C —— 无延迟族对照
    # ============================================================
    def exp_c_no_latency(self) -> dict:
        """
        ③ 剔除整个 18 维延迟族，只留节律 6 + 交互 8。

        七组消融全都含完整延迟族，本组是唯一真正把它拿掉的对照 ——
        「二分类失效与特征口径无关」这一论断要靠它才站得住。

        Returns:
            dict
        """
        log('=' * 62, 'info')
        log('实验 C — 无延迟族对照（破「七组互相独立」）', 'step')
        log('=' * 62, 'info')
        nolat = [c for c in FE.BEHAVIOR_FEATURES if c not in FE.F01_LATENCY]
        r = self._fit('c_nolatency', self.df, nolat)
        if r:
            log(f'  无延迟族({r["n_nominal"]} 维): macro-F1={r["macro_f1"]:.4f} '
                f'agent→human={r["as_human_pct"]}%', 'info')
        return r

    # ============================================================
    #  ④ 报告
    # ============================================================
    def report(self) -> str:
        a = self.out.get('exp_a') or {}
        b = self.out.get('exp_b') or {}
        c = self.out.get('exp_c') or {}
        L = [f'# 稳健性检验报告（{self.chain}）\n',
             '> 由 `src/verify/robustness_checks.py` 自动生成。',
             '> 与 `TRAIN_REPORT` 同数据、同管线、同种子，指标可直接对比。\n']

        # ---- 表 A ----
        L.append('\n## 表 A 捷径单独成模 —— 消融 Δ 为什么不能读成「无贡献」\n')
        L.append('七组消融里，剔除任一类捷径的 Δ 都只有 0.0001~0.0002。'
                 '但让每条捷径**单独**成模：\n')
        L.append('| 对照 | 名义维数 | 入模维数 | macro-F1 | agent 召回 |')
        L.append('|---|---|---|---|---|')
        for k, r in a.items():
            L.append(f"| {r['desc']} | {r['n_nominal']} | {r['n_used']} | "
                     f"**{r['macro_f1']:.4f}** | "
                     f"{r['recall'].get('agent', float('nan')):.4f} |")
        m16 = a.get('miss16', {})
        g1 = a.get('gas_avail1', {})
        if m16 and g1:
            L.append(
                f"\n**读法**：缺失模式一族单独成模即达 macro-F1 "
                f"{m16['macro_f1']:.4f}、agent 召回 "
                f"{m16['recall'].get('agent', 0):.4f}；"
                f"单个 `f_gas_avail_ratio` 的 agent 召回也有 "
                f"{g1['recall'].get('agent', 0):.4f}。\n"
                '三条捷径各自都能独立锁死 agent ⇒ 拿掉一条，剩下的立刻补位，'
                'Δ 自然≈0。\n'
                '**所以 Δ≈0 证明的是「捷径之间高度冗余」，'
                '而不是「捷径没有贡献」。**\n'
                '主结论请引用 behavior 集的**绝对值**，不要引用 Δ。\n')

        # ---- 表 B ----
        if b.get('baseline') and b.get('censored'):
            bb, bc = b['baseline'], b['censored']
            L.append('\n## 表 B Δt 截断对照 —— 采集不对称的定量影响\n')
            L.append(
                f"bot 组按 {config.BOT_COLLECT_SEGMENTS} 段 × "
                f"{config.BOT_COLLECT_SEG_BLOCKS:,} 块采集，段内 Δt 物理上限约 "
                f"**{b['cap_s']:,.0f} 秒**；agent/human 全窗口连续，上限是整个窗口。\n"
                '把三组 Δt 一律 winsorize 到该上限（不丢地址，样本量不变）后重训 '
                'behavior 集：\n')
            L.append('| 口径 | 样本 | macro-F1 | agent 召回 | bot 召回 '
                     '| human 召回 | 二分类 agent→human |')
            L.append('|---|---|---|---|---|---|---|')
            for tag, r in (('基线（论文口径）', bb),
                           (f"Δt winsorize 至 {b['cap_s']:,.0f}s", bc)):
                rc = r['recall']
                L.append(
                    f"| {tag} | {r['n_samples']:,} | **{r['macro_f1']:.4f}** | "
                    f"{rc.get('agent', 0):.3f} | {rc.get('bot', 0):.3f} | "
                    f"{rc.get('human', 0):.3f} | {r['as_human_pct']}% |")
            d1 = bb['macro_f1'] - bc['macro_f1']
            d2 = bb['as_human_pct'] - bc['as_human_pct']
            L.append(
                f"\n**读法**：主结论在截断后仍达 **{bc['macro_f1']:.4f}**"
                f"（较基线 −{d1:.4f}），远超 {config.GONOGO_F1_GO} 判定线 ⇒ "
                '「行为指纹存在」这一核心论点不依赖采集不对称。\n'
                f"但二分类漏检率从 {bb['as_human_pct']}% 降到 "
                f"{bc['as_human_pct']}%（−{d2:.1f}pp）⇒ "
                '**表 5.5 那个具体区间对采集口径敏感，论文不应把它当成硬边界**，'
                '应表述为「多数 agent 被归并为 human」这一方向性结论。\n')

        # ---- 表 C ----
        if c:
            L.append('\n## 表 C 无延迟族对照 —— 七组消融并不互相独立\n')
            L.append(
                'model / latency / nometa / nowin / clean / behavior / timing '
                '**七组全部包含完整的 18 维延迟族**，消融只动延迟族以外的部分。\n'
                '所以「七组无一例外」不是七个独立证据。'
                '真正的对照是把延迟族整族拿掉：\n')
            L.append('| 对照 | 名义维数 | 入模维数 | macro-F1 | agent 召回 '
                     '| 二分类 agent→human |')
            L.append('|---|---|---|---|---|---|')
            L.append(f"| behavior − 延迟族（节律 6 + 交互 8） | "
                     f"{c['n_nominal']} | {c['n_used']} | "
                     f"**{c['macro_f1']:.4f}** | "
                     f"{c['recall'].get('agent', 0):.3f} | "
                     f"{c['as_human_pct']}% |")
            h = c['as_human_pct']
            keep = ('仍占多数' if h > 50 else
                    '已不再占多数（接近五五开）' if h > 40 else '已翻转到 bot 一侧')
            L.append(
                f"\n**读法（结论比原论文更保守，必须如实写）**：拿掉延迟族后，"
                f"agent 被判为 human 的比例降到 **{h}%**，{keep}。\n"
                f"对比七组共享延迟族时的 56.2%~67.6%，可以确定：\n"
                '- ✅ **站得住的部分**：无论哪种特征口径，agent 都被强行拆到'
                'human/bot 两侧，没有一侧能正确容纳它 —— '
                '「二分标签空间里没有 agent 的位置」这一结构性论断成立；\n'
                '- ✅ 被判为 human 的比例始终**远高于零**，且与 Web 域报道的 '
                '34.5%~39.1% 处于同一量级甚至更高；\n'
                '- ⚠️ **站不住的部分**：「漏检为 human 的比例一致落在 '
                '56.2%~67.6%」与「失效方向与特征可得性无关」。'
                f'这个多数方向主要由延迟族驱动 —— 整族拿掉后就降到 {h}%，'
                '再叠加表 B 的采集口径敏感性，说明该区间不是硬边界。\n\n'
                '⇒ 论文应把结论收敛为「**agent 在二分框架下必然被错误归并，'
                '且相当大比例被归并到风控上最危险的 human 一侧**」，'
                '并同时报告本表与表 B 作为边界条件，'
                '而不是把七个共享内核的集合当成七个独立证据。\n')

        L.append('\n---\n')
        L.append('## 建议的论文写法\n')
        L.append('1. 5.6 节删掉「Δ=+0.0002 说明协议形态捷径贡献近乎为零」，'
                 '改为「三类捷径互为冗余，逐项消融在 0.99 以上的天花板区间'
                 '没有分辨率（表 A）；判别力的证据是 behavior 集的绝对值」。')
        L.append('2. 5.7 节删掉「七组无一例外 ⇒ 失效与特征可得性无关」这一推论'
                 '（七组共享同一个延迟族内核，不构成七个独立证据），'
                 '改为「agent 在二分框架下必然被错误归并，且相当大比例落到 '
                 'human 一侧」，并把表 B、表 C 作为边界条件一并报告。')
        L.append('3. 摘要、1.4 节、5.9 节、6.1 节里所有'
                 '「56.2%~67.6%」「与特征口径无关」的表述同步改口径。')
        L.append('4. 5.8 节局限性第 (3) 条补上表 B 的定量结果，'
                 '不要只写「可能轻微影响长周期节律特征」。')
        return '\n'.join(L)

    # ============================================================
    #  ⑤ 主流程
    # ============================================================
    def run(self) -> dict:
        self.load()
        self.out['exp_a'] = self.exp_a_shortcut_alone()
        self.out['exp_b'] = self.exp_b_dt_censor()
        self.out['exp_c'] = self.exp_c_no_latency()

        md = self.report()
        (config.FIG_DIR / f'ROBUSTNESS_{self.chain}.md').write_text(
            md, encoding='utf-8')
        (config.FIG_DIR / f'ROBUSTNESS_STATS_{self.chain}.json').write_text(
            json.dumps(self.out, ensure_ascii=False, indent=2), encoding='utf-8')
        log(f'报告 → figures/ROBUSTNESS_{self.chain}.md', 'ok')
        log(f'实测值 → figures/ROBUSTNESS_STATS_{self.chain}.json', 'ok')
        return self.out


def main():
    ap = argparse.ArgumentParser(description='稳健性检验（三组对照）')
    ap.add_argument('--chain', default=config.PRIMARY_CHAIN,
                    choices=list(config.CHAINS))
    args = ap.parse_args()
    RobustnessChecks(chain=args.chain).run()


if __name__ == '__main__':
    main()
