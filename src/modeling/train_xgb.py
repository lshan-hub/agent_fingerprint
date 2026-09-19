#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：train_xgb.py
@Description:
    三分类训练与预测入口 —— 串起「特征表 → XgbClfModel → 判定书」。

    ============================================================
    一、 三套特征集的对照实验（论文必做）
    ============================================================
    同一份数据跑三次，报告差值:

      ① model      全部 57 特征                  —— 主模型（上界）
      ② latency    仅延迟族 18 个                —— 单独验证 H1
      ③ nometa     剔除 3 个元特征               —— 去「协议形态」捷径
      ④ nowin      剔除 6 个窗口依赖特征          —— 去「采集窗口」捷径
      ⑤ clean      ③+④                          —— 严对照
      ⑥ behavior   再去 gas/nonce/value 三族      —— ★★★主结论（去「缺失模式」捷径）
      ⑦ timing     只剩延迟+节律                 —— 结论下界（去「交互对象」捷径）

      🔴 为什么要一路收紧到 ⑥⑦:
        判别力可以从四条「捷径」漏进来，每条都会让结论退化成同义反复:
          捷径1 协议形态 —— 元特征直接编码「走不走元交易」
          捷径2 采集窗口 —— 窗口不等长时 entropy_dow 恒为 0 vs 恒为正
          捷径3 缺失模式 —— XGBoost 把「有没有值」当分裂方向，
                            agent 组 nonce 缺失 90%，等价于捷径1 换个形式
          捷径4 交互对象 —— agent 98% 活动是 USDC 事件，f_to_entropy 被钉死为 0
        ⑥ 堵掉 1-3，⑦ 连 4 一起堵。论文主结论引用 ⑥，下界引用 ⑦。

      🔴 为什么 ③ 最关键:
        f_txfrom_ratio / f_gas_avail_ratio / f_kind_entropy 描述的是
        **数据可得性**而非行为差异。x402 付款方走元交易（Facilitator 代付）
        ⇒ 结构性没有 gasPrice/nonce，这三个特征能几乎完美地识别出 agent。
        · 若不剔除，结论会退化成「agent 用了不同的支付协议」这种同义反复
        · 剔除后仍能分开，才能支撑「agent 有独特行为指纹」的论点
        ⇒ ①③ 的差值就是「协议形态贡献了多少判别力」，必须写进论文。

    ============================================================
    二、 判定口径（沿用 go/no-go 三条线）
    ============================================================
      1) 三分类 macro-F1        阈值 config.GONOGO_F1_GO(0.70) / _CAUTION(0.55)
      2) agent 类召回率          阈值 config.GONOGO_AGENT_RECALL_GO(0.60) / _MIN(0.40)
      3) ★ 二分类对照的漏检率    对标 Web 域 39.1% / 34.5%

    ============================================================
    三、 调用方式
    ============================================================
      # 训练（三套特征集全跑）
      python3 src/modeling/train_xgb.py

      # 只跑主特征集
      python3 src/modeling/train_xgb.py --featureset model

      # 合成数据烟测
      python3 src/modeling/train_xgb.py --synthetic

      # 预测（对新地址打分）
      python3 src/modeling/train_xgb.py --predict data/features/features_base.csv

    ============================================================
    四、 落盘
    ============================================================
      data/model/{tag}_{featureset}/
        xgb_clf_full.pkl / config.pkl / metrics.json / feature_report.csv
      figures/TRAIN_REPORT_{tag}.md    —— 三套特征集的对照报告
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
from src.feature.f_ensemble import FeatureEnsemble  # noqa: E402
from src.modeling.XgbClf import XgbClfModel  # noqa: E402


# ================================================================
#  TrainPipeline — 三套特征集的对照训练
# ================================================================
class TrainPipeline(object):
    """
    训练管线：同一份数据在三套特征集上各训一次，输出对照报告。

    核心函数:
      ① load()            载入特征表
      ② train_one(name)   训练单套特征集
      ③ run()             三套全跑 + 对照报告
      ④ predict(path, df) 用已训模型打分

    Args:
        tag: 数据标签（链名或 'synthetic'）
    """

    FEATURE_SETS = {
        'model':   ('全部 57 特征（主模型）', FeatureEnsemble.MODEL_FEATURES),
        'latency': ('仅延迟族 18 特征（验证 H1）', FeatureEnsemble.LATENCY_ONLY_FEATURES),
        'nometa':  ('剔除元特征（★关键对照·协议形态）', FeatureEnsemble.NO_META_FEATURES),
        'nowin':   ('剔除窗口依赖特征（对照·采集设计）', FeatureEnsemble.NO_WINDOW_FEATURES),
        'clean':   ('元特征+窗口特征全剔（★★严对照）', FeatureEnsemble.CLEAN_FEATURES),
        'behavior': ('纯行为·三组均可算（★★★主结论）',
                     FeatureEnsemble.BEHAVIOR_FEATURES),
        'timing':   ('纯时间行为·最保守（结论下界）',
                     FeatureEnsemble.TIMING_FEATURES),
    }

    def __init__(self, tag: str = None):
        """构造: 存数据标签。"""
        self.tag = tag or config.PRIMARY_CHAIN
        self.df = pd.DataFrame()
        self.results = {}

    # ============================================================
    #  ① 载入
    # ============================================================
    def load(self) -> pd.DataFrame:
        """载入 data/features/features_{tag}.csv。"""
        p = config.DATA_FEAT / f'features_{self.tag}.csv'
        if not p.exists():
            log(f'找不到 {p}', 'err')
            log('先跑：python3 src/feature/f_ensemble.py', 'info')
            sys.exit(1)
        self.df = pd.read_csv(p)
        log(f'载入 {len(self.df)} 行 × {len(self.df.columns)} 列', 'ok')
        log(f'类别分布: {dict(self.df["group"].value_counts())}', 'info')
        return self.df

    # ============================================================
    #  ② 训练单套
    # ============================================================
    def train_one(self, name: str) -> dict:
        """
        训练单套特征集

        Args:
            name: 'model' / 'latency' / 'nometa'
        Returns:
            dict: metrics
        """
        desc, feats = self.FEATURE_SETS[name]
        use = [f for f in feats if f in self.df.columns]
        keep = ['address', 'group'] + use
        sub = self.df[[c for c in keep if c in self.df.columns]].copy()

        log('', 'info')
        log('#' * 62, 'info')
        log(f'# 特征集 [{name}] {desc} —— {len(use)} 个特征', 'step')
        log('#' * 62, 'info')

        save = config.DATA_DIR / 'model' / f'{self.tag}_{name}'
        m = XgbClfModel(df_train=sub, save_path=save,
                        top_n=min(40, len(use)), verbose=True)
        try:
            metrics = m.train()
        except Exception as e:
            log(f'[{name}] 训练失败: {type(e).__name__}: {e}', 'err')
            return {}
        metrics['feature_set'] = name
        metrics['save_path'] = str(save)
        self.results[name] = metrics
        return metrics

    # ============================================================
    #  ③ 三套全跑 + 对照报告
    # ============================================================
    def _miss_note(self) -> str:
        """现算「缺失敏感特征对 agent 组的缺失率区间」，供报告引用。

        口径与图 4.3 / 特征说明.md 第九节一致：只统计 clean→behavior 之间
        被剔除的 16 维里真正会缺失的成员（f_value_zero_ratio 恒可算，不计）。
        """
        removed = [c for c in FeatureEnsemble.CLEAN_FEATURES
                   if c not in FeatureEnsemble.BEHAVIOR_FEATURES
                   and c in self.df.columns]
        if not removed or 'group' not in self.df.columns:
            return '它们对 agent 组结构性缺失'
        rate = self.df.groupby('group')[removed].apply(lambda x: x.isna().mean())
        sens = [c for c in removed if (rate[c].max() - rate[c].min()) > 0.30]
        if not sens or 'agent' not in rate.index:
            return '它们对 agent 组结构性缺失'
        ag = rate.loc['agent', sens] * 100
        return (f'其中 {len(sens)}/{len(removed)} 维缺失敏感，'
                f'对 agent 组缺失率 {ag.min():.1f}%~{ag.max():.1f}%')

    def report(self) -> str:
        """生成三套特征集的对照报告（Markdown）。"""
        L = [f'# 三分类训练报告（{self.tag}）\n']
        vc = self.df['group'].value_counts()
        L.append(f'样本: {len(self.df)} 个地址 | 类别分布: {dict(vc)}\n')

        # ---- 论文表 5.1: 超参数逐组扫描过程 ----
        # 只报告最终 Top-1 超参是不够的 —— 审稿人要看搜索路径:
        # 扫了哪些候选、每轮选了什么、分数抬升多少。
        rm0 = self.results.get('model') or next(
            (r for r in self.results.values() if r), None)
        rounds = (rm0 or {}).get('tuning_rounds') or []
        if rounds:
            L.append('\n## 0. 超参数逐组扫描（论文表 5.1）\n')
            L.append('分组扫描而非全组合网格：每轮只调一组相关超参，'
                     '将当轮最优并入基线后进入下一轮，避免组合爆炸。\n')
            L.append('每轮以分层交叉验证的 macro-F1 为准则。\n')
            L.append('| 轮次 | 调参组 | 候选取值 | 组合数 | 当轮选定 | CV macro-F1 | 增量 |')
            L.append('|---|---|---|---|---|---|---|')
            for r in rounds:
                d = ('—' if r.get('delta') is None else f"{r['delta']:+.4f}")
                L.append(f"| {r['round']} | {r['group']} | {r['candidates']} | "
                         f"{r['n_combos']} | {r['chosen']} | "
                         f"{r['cv_macro_f1']:.4f} | {d} |")
            bp = (rm0 or {}).get('best_params') or {}
            if bp:
                L.append('\n**最终 Top-1 超参**: '
                         + ', '.join(f'`{k}={v}`' for k, v in bp.items()) + '\n')

        L.append(f'\n## 1. {len(self.FEATURE_SETS)} 套特征集对照（从宽到严，越往下越保守）\n')
        L.append('| 特征集 | 说明 | 特征数 | OOF macro-F1 | agent 召回 | bot 召回 | human 召回 |')
        L.append('|---|---|---|---|---|---|---|')
        for n, (desc, _) in self.FEATURE_SETS.items():
            r = self.results.get(n)
            if not r:
                L.append(f'| {n} | {desc} | — | 训练失败 | — | — | — |')
                continue
            rec = r.get('oof_recall', {})
            L.append(f"| **{n}** | {desc} | {r['n_features']} | "
                     f"**{r['oof_macro_f1']:.4f}** | "
                     f"{rec.get('agent', float('nan')):.3f} | "
                     f"{rec.get('bot', float('nan')):.3f} | "
                     f"{rec.get('human', float('nan')):.3f} |")

        # 🔴 关键差值
        rm, rn = self.results.get('model'), self.results.get('nometa')
        if rm and rn:
            d = rm['oof_macro_f1'] - rn['oof_macro_f1']
            L.append(f'\n### 🔴 元特征贡献度 = {d:+.4f}\n')
            L.append(f"全部特征 macro-F1 {rm['oof_macro_f1']:.4f} vs "
                     f"剔除元特征 {rn['oof_macro_f1']:.4f}\n")
            # 🔴 2026-09-13 修正：Δ≈0 不等于「捷径没贡献」。
            #    model 集本身已在 0.9997 的天花板上，多条捷径互为冗余，
            #    拿掉一条另一条立刻补位 ⇒ 消融差值在此区间没有分辨率。
            #    正确的读法要配合「捷径单独成模」的绝对值一起看。
            if rm['oof_macro_f1'] >= 0.99 and abs(d) < 0.01:
                L.append('> ⚠️ **不要把这个 Δ 读成「协议形态没贡献」。**\n'
                         f"> model 集已达 {rm['oof_macro_f1']:.4f}，逼近天花板；"
                         '多条捷径互为冗余，拿掉一条另一条会立刻补位，\n'
                         '> 所以 Δ≈0 说明的是**冗余**，不是**无贡献**。\n'
                         '> ⇒ 判据请改用两条：① 捷径**单独成模**能到多少'
                         '（见 `src/verify/robustness_checks.py`）；\n'
                         '> ② behavior 集在剔除全部捷径后的**绝对值**。\n')
            elif abs(d) < 0.03:
                L.append('> ✅ 差值很小，且 model 集未触及天花板 ⇒ '
                         '协议形态捷径的边际贡献有限。\n'
                         '> 仍建议配合「捷径单独成模」的绝对值交叉验证。\n')
            elif d > 0.10:
                L.append('> 🔴 **差值很大 ⇒ 判别力主要来自「数据可得性」而非行为。**\n'
                         '> 结论会退化成「agent 用了不同的支付协议」这种同义反复，\n'
                         '> 论文必须以 nometa 的结果为准，并明确讨论这个问题。\n')
            else:
                L.append('> ⚠️ 差值中等 ⇒ 协议形态贡献了部分判别力，论文需同时报告两套结果。\n')

        # 🔴 窗口依赖差值 —— human 组窗口 ×12，防「靠窗口长度作弊」
        rw = self.results.get('nowin')
        if rm and rw:
            d = rm['oof_macro_f1'] - rw['oof_macro_f1']
            L.append(f'\n### 🔴 窗口依赖特征贡献度 = {d:+.4f}\n')
            L.append(f"全部特征 {rm['oof_macro_f1']:.4f} vs 剔除窗口依赖特征 "
                     f"{rw['oof_macro_f1']:.4f}\n")
            L.append('> 消融的 6 个特征: 3 个绝对跨度（span/active_days/'
                     'longest_silence）+ 3 个周期覆盖度（entropy_dow/'
                     'entropy_24h/active_hours）。\n'
                     '> 后者尤其危险: 窗口不足一周时 dow 熵恒为 0，'
                     '模型可靠「熵>0 ⇒ human」作弊。\n'
                     '> 现三组窗口已统一为 302,400 块（7 天），此项验证是否还有残留泄漏。\n')
            if rm['oof_macro_f1'] >= 0.99 and abs(d) < 0.01:
                L.append('> ⚠️ 同上：天花板区间内的 Δ 无分辨率，不能读成'
                         '「采集窗口没贡献」。\n'
                         '> 实测这 6 维**单独成模**即可达 0.90 量级 —— '
                         '它们有强判别力，只是与其他捷径冗余。\n')
            elif abs(d) < 0.03:
                L.append('> ✅ 差值很小 ⇒ 归一化有效，窗口设计的边际贡献有限。\n')
            else:
                L.append('> 🔴 **差值偏大 ⇒ 仍有窗口泄漏，论文应以 clean 集为准。**\n')

        rc = self.results.get('clean')
        if rm and rc:
            L.append(f"\n### ★★ 严对照（clean）macro-F1 = "
                     f"{rc['oof_macro_f1']:.4f}\n")
            # 🔴 缺失率现算，不写死（此前硬编码的 "68~100%" 是旧管线遗留值，
            #    并被论文 4.1 节引用成 "79%~98%"，两处都与实测对不上）
            L.append('> 同时剔除元特征与窗口依赖特征。\n'
                     f'> ⚠️ 但仍含 gas/nonce/value 三族 —— {self._miss_note()}，'
                     'XGBoost 可靠「有没有值」间接识别组别。\n')

        rb = self.results.get('behavior')
        if rm and rb:
            d = rm['oof_macro_f1'] - rb['oof_macro_f1']
            L.append(f"\n### ★★★ 主结论：纯行为集（behavior）macro-F1 = "
                     f"{rb['oof_macro_f1']:.4f}\n")
            L.append(f"仅用 {rb['n_features']} 个三组均可算的特征"
                     f"（延迟 + 节律 + 交互），对全特征差值 {d:+.4f}。\n")
            L.append('> 这一栏排除了三类捷径: ①协议形态（元特征）'
                     '②采集窗口（跨度特征）③缺失模式（gas/nonce/value 族）。\n'
                     '> **论文的主结论应以这一栏的绝对值为准** —— '
                     '它是唯一只能用「行为本身不同」来解释的结果。\n'
                     f'> ⚠️ 上面那个差值 {d:+.4f} 只说明捷径与行为信号高度冗余，'
                     '**不能**被读成「三类捷径合计只贡献 '
                     f'{d * 100:.1f} 个百分点」——\n'
                     '> 在 0.99 以上的天花板区间，消融差值没有分辨率。\n')
            if rb['oof_macro_f1'] >= 0.70:
                L.append('> ✅ 纯行为特征仍能三分类 ⇒ '
                         '「链上 AI agent 存在可测的行为指纹」这一核心论点成立。\n')
            else:
                L.append('> 🔴 纯行为特征分不开 ⇒ 前面几栏的高分主要来自'
                         '协议/采集差异，论文应转向可观测性边界的负结果。\n')

        rl = self.results.get('latency')
        if rm and rl:
            d = rm['oof_macro_f1'] - rl['oof_macro_f1']
            L.append(f"\n### 延迟族单独贡献\n\n仅用 {rl['n_features']} 个延迟特征达到 "
                     f"macro-F1 {rl['oof_macro_f1']:.4f}，全部特征 "
                     f"{rm['oof_macro_f1']:.4f}，差值 {d:+.4f}。\n")
            L.append('> 差值越小，说明 H1「响应延迟是核心信号」越成立。\n')

        # 二分类对照
        for n in ('model', 'nometa'):
            r = self.results.get(n) or {}
            b = r.get('binary_control') or {}
            if not b:
                continue
            L.append(f'\n## 2. ★ 二分类对照实验（{n}）\n')
            L.append('用 human/bot 二分类器去分 agent 地址：\n')
            L.append('| 结果 | 占比 |')
            L.append('|---|---|')
            L.append(f"| agent → 判为 **human** | **{b['as_human_pct']:.1f}%** |")
            L.append(f"| agent → 判为 **bot** | {b['as_bot_pct']:.1f}% |")
            L.append(f"\n对标 Web 域 (arXiv 2607.26935): "
                     f"MLP 漏检 {b['web_domain_mlp']}% / SAINT 漏检 "
                     f"{b['web_domain_saint']}%\n")
            break

        # go/no-go 判定 —— 以最严的 behavior 集为准
        # （退化顺序 behavior > clean > nometa > model，取跑过的最严一档）
        r = rb or rc or rn or rm
        base = ('behavior' if rb else 'clean' if rc
                else 'nometa' if rn else 'model')
        if r:
            f1 = r['oof_macro_f1']
            ra = (r.get('oof_recall') or {}).get('agent', 0)
            if f1 >= config.GONOGO_F1_GO and ra >= config.GONOGO_AGENT_RECALL_GO:
                flag = '🟢 GO —— 三分类信号成立，全速推进'
            elif f1 >= config.GONOGO_F1_CAUTION and ra >= config.GONOGO_AGENT_RECALL_MIN:
                flag = '🟡 CAUTION —— 信号存在但不够强，扩充真值后复测'
            else:
                flag = '🔴 NO-GO —— 转向可观测性边界的负结果论文，或改课题二'
            L.append(f'\n## 3. go/no-go 判定（以 {base} 为准）\n\n**{flag}**\n')
            L.append(f'\nmacro-F1 = {f1:.4f}（线 {config.GONOGO_F1_GO}）'
                     f' | agent 召回 = {ra:.4f}（线 '
                     f'{config.GONOGO_AGENT_RECALL_GO}）\n')

        L.append('\n## 4. 已知局限（写论文时必须交代）\n')
        L.append('- **Δt 是代理量**，不是真正的 event-response latency；'
                 '亚秒差异在链上永远不可见（时间戳整秒）')
        L.append('- **三组的链上足迹形态不同**：MEV bot 几乎全是 tx_from，'
                 'x402 付款方几乎全是 event，Olas Safe 是被调用 —— '
                 'gas/nonce 特征对 agent 组结构性缺失')
        L.append('- **人类样本是弱先验**：CEX 提现不等于自然人，'
                 '需做 5-30% 标签噪声敏感性分析；'
                 f'且 human 组另有一条只对本组生效的活跃度上限'
                 f'（config.HUMAN_MAX_ACTS_WINDOW='
                 f'{config.HUMAN_MAX_ACTS_WINDOW:,} 笔/全窗口），'
                 '会压低该组的活动频率，论文必须披露')
        L.append(f'- **🔴 三组 Δt 的观测支撑不等长**：bot 组按 '
                 f'{config.BOT_COLLECT_SEGMENTS} 段 × '
                 f'{config.BOT_COLLECT_SEG_BLOCKS:,} 块采集，段内 Δt 物理上限约 '
                 f'{config.BOT_COLLECT_SEG_BLOCKS * 2:,} 秒；'
                 'agent/human 全窗口连续，Δt 上限是整个窗口。'
                 '延迟族的尾部统计量（p75/mean/std/iqr）因此不完全可比 —— '
                 '定量影响见 `src/verify/robustness_checks.py` 的 Δt 截断对照')
        L.append('- **消融 Δ 在指标饱和时不可解读**：model 集已达 0.9997，'
                 '各捷径互为冗余，拿掉一条另一条立刻补位 ⇒ Δ≈0 说明的是'
                 '「冗余」而非「无贡献」。论文主结论请引用 behavior 集的'
                 '**绝对值**，不要引用 Δ（见 robustness_checks 的捷径单独成模表）')
        return '\n'.join(L)

    def run(self, only: str = None) -> dict:
        """
        ③ 三套特征集全跑 + 对照报告

        Args:
            only: 只跑某一套（'model'/'latency'/'nometa'）
        Returns:
            dict: {featureset: metrics}
        """
        self.load()
        names = [only] if only else list(self.FEATURE_SETS)
        for n in names:
            self.train_one(n)

        md = self.report()
        out = config.FIG_DIR / f'TRAIN_REPORT_{self.tag}.md'
        out.write_text(md, encoding='utf-8')
        print('\n' + '=' * 66)
        print(md.split('\n## 4.')[0])
        print('=' * 66)
        log(f'完整报告：{out}', 'ok')
        return self.results

    # ============================================================
    #  ④ 预测
    # ============================================================
    @staticmethod
    def predict(model_dir: str, feature_csv: str, out_csv: str = None) -> pd.DataFrame:
        """
        ④ 用已训模型对新地址打分

        Args:
            model_dir:   训练时的落盘目录
            feature_csv: 特征表 CSV
            out_csv:     输出路径；缺省 data/features/pred_{stem}.csv
        Returns:
            pd.DataFrame: [address, pred_label, p_human, p_bot, p_agent]
        """
        df = pd.read_csv(feature_csv)
        conf, _ = XgbClfModel._load_bundle(model_dir)
        classes = conf['classes']

        proba = XgbClfModel.pred(model_dir, df)
        label = XgbClfModel.pred_label(model_dir, df)

        out = pd.DataFrame({'address': df.get('address', pd.Series(range(len(df)))),
                            'pred_label': label})
        for i, c in enumerate(classes):
            out[f'p_{c}'] = proba[:, i]
        in_sample = False
        if 'group' in df.columns:
            out['true_label'] = df['group']
            acc = float((out['pred_label'] == out['true_label']).mean())
            # 🔴 这里的模型是 Phase 4 在**全量样本**上重训的线上模型。
            #    若 feature_csv 就是训练用的那张特征宽表，这个 accuracy 是
            #    **训练集内精度**，必然接近 1.0，不是泛化性能，绝不可入论文。
            #    论文的性能数字一律取 metrics.json 的 OOF 指标。
            in_sample = True
            log(f'⚠️ 训练集内 accuracy = {acc:.4f}'
                f'（Phase 4 全量重训模型对自己的训练数据打分，'
                f'不是泛化性能；泛化指标见 metrics.json 的 oof_macro_f1）', 'warn')

        p = Path(out_csv or (config.DATA_FEAT /
                             f'pred_{Path(feature_csv).stem}.csv'))
        if in_sample:
            # 把警告写进文件头，防止下游把这张表当评估结果引用
            with p.open('w', newline='', encoding='utf-8') as f:
                f.write('# ⚠️ 冒烟测试产物，非评估结果。\n'
                        '# 模型 = Phase 4 全量重训的线上模型；输入 = 它自己的'
                        '训练数据 ⇒ 本表 accuracy 必然接近 1.0（训练集内精度）。\n'
                        '# 论文性能一律引用 data/model/*/metrics.json 的 OOF 指标。\n'
                        '# 本表的正当用途：排查错例 / 校验预测接口是否跑得通。\n')
                out.to_csv(f, index=False)
        else:
            out.to_csv(p, index=False)
        log(f'预测结果写入 {p}（{len(out)} 行）', 'ok')
        print(out.head(10).to_string(index=False))
        return out


def main():
    ap = argparse.ArgumentParser(description='三分类训练与预测')
    ap.add_argument('--chain', default=config.PRIMARY_CHAIN,
                    choices=list(config.CHAINS))
    ap.add_argument('--synthetic', action='store_true', help='使用合成数据')
    ap.add_argument('--featureset', default=None,
                    choices=list(TrainPipeline.FEATURE_SETS),
                    help='只跑某一套特征集（缺省三套全跑）')
    ap.add_argument('--predict', default=None,
                    help='预测模式：给出特征表 CSV 路径')
    ap.add_argument('--model-dir', default=None, help='预测用的模型目录')
    args = ap.parse_args()

    tag = 'synthetic' if args.synthetic else args.chain
    if args.predict:
        md = args.model_dir or str(config.DATA_DIR / 'model' / f'{tag}_nometa')
        TrainPipeline.predict(md, args.predict)
    else:
        TrainPipeline(tag=tag).run(only=args.featureset)


if __name__ == '__main__':
    main()
