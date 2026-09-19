#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_human.py
@Description:
    人类样本候选采集（Base 链上直接反推）—— 三组真值中最难的一组。

    ============================================================
    一、 为什么需要这个模块
    ============================================================
    人类样本没有免费的单一来源:
      · Coinglass 的 CEX 提币数据 **只有 Ethereum ERC-20**，而主战场是 Base
        （以太坊 12s 出块 ⇒ agent 带 2-10s 在物理上不存在，H1 不可检验）
      · Dune 的 labels.cex 要烧 credits，且表名历年改过多次

    ⇒ 本模块直接在 Base 链上用**行为定义**反推 CEX 热钱包，再取其提币接收方。
      好处: 免费、主战场原生、不依赖任何外部标签表。

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — CEX 提币接收方 (Base 链上反推) ─────────────────────────────┐
    │  接口:  SQD stream，两阶段扫描                                            │
    │           阶段1: 扫全链交易 → 按 from 聚合，找「散播型」地址 = CEX 热钱包 │
    │           阶段2: 取这些热钱包的接收方 = 人类样本候选                       │
    │  函数:  fetch_humans() (内部 _find_hot_wallets / _collect_recipients)     │
    │  更新频率: 实时 │ API KEY: 否 │ 历史范围: 任意窗口                        │  ★必填
    │  产出: data/addresses/human_cex.csv                                       │
    └───────────────────────────────────────────────────────────────────────────┘

      ★ CEX 热钱包的行为定义（不依赖标签表）:
          · 单地址向**大量不同接收方**转账（散播型，n_recipients 高）
          · 且接收方**重复率低**（不是在跟固定几个合约交互）
        这精准刻画了「交易所给用户提币」的形态。

      链上原始返回     → 统一字段        → 说明
      tx.to           → address         提币接收方（★人类样本候选）
      tx.from         → source_cex      热钱包地址（写进 note 溯源）
      [按地址计数]     → n_from_cex      收到几次提币

    ============================================================
    三、 清洗逻辑 & 🔴 反循环设计
    ============================================================
      1. 热钱包判据: n_recipients >= MIN_RECIPIENTS 且 唯一接收方占比 >= 0.8
      2. 接收方必须是 **EOA**（eth_getCode 为空）—— 排除合约，合约不是人
      3. 排除已在 agent / bot 组的地址（避免标签冲突）
      4. 🔴 **绝不用「昼夜节律」筛选** ——
         这是文档反复警告的循环风险: 用昼夜节律筛人类、又用 entropy_24h 当特征
         = 用答案筛样本，会人为抬高模型性能。
         本模块只用「收到 CEX 提币 + 是 EOA」两条与节律无关的判据。
         ⇒ entropy_24h / night_ratio 因此是**独立证据**，可放心进模型。

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🥉 三组真值中「人类」组的来源。真值质量 ★★（弱先验）。
      🔴 论文必须交代: CEX 提现只是弱先验不是证明 ——
        机构、做市商、甚至 bot 都可能从交易所提币。
        必须做 5-30% 标签噪声敏感性分析。
'''

import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils._base import BaseFetcher, RpcPool, log  # noqa: E402
from src.utils.common import read_address_csv  # noqa: E402
from src.utils.sqd_client import SQDClient  # noqa: E402


# ================================================================
#  HumanDataSource — Base 链上反推 CEX 提币接收方
# ================================================================
class HumanDataSource(object):
    """
    人类样本候选采集与管理（Base 链上反推，不依赖外部标签表）

    核心函数 (共 1 个):
      ① fetch_humans(blocks, ...)  两阶段扫描 → CEX 提币接收方

    辅助函数:
      - _find_hot_wallets()    阶段1: 按散播行为反推 CEX 热钱包
      - _collect_recipients()  阶段2: 取热钱包的接收方
      - _filter_eoa()          只保留 EOA（合约不是人）

    校验函数: validate_humans

    Args:
        lo / hi: 区块窗口；缺省 config.MODEL_WINDOW_*
    """

    # 热钱包判据（清洗逻辑 1）
    MIN_RECIPIENTS = 40         # 至少向多少个不同地址转过账
    MIN_UNIQUE_RATIO = 0.8      # 唯一接收方占比（排除跟固定合约反复交互的）
    # 🔴 阶段3 活跃度门槛：探测窗口内自发交易数下限。
    #    实测不加此筛选时人类组 100% 地址活动不足被剔除（见 _filter_active 文档）
    MIN_ACTIVE_TX = 3
    # 🔴 上限同样必要：实测候选中出现过 184,394 笔/33 小时（≈3 笔/块）的地址，
    #    那是混进来的 bot/热钱包 EOA，不是人。人类 33 小时内几十笔已是重度用户。
    MAX_ACTIVE_TX = 200
    ACTIVE_PROBE_BLOCKS = 60000  # 活跃度探测窗口（Base 2s ⇒ ≈33 小时）
    MAX_HOT_WALLETS = 20        # 最多取几个热钱包

    def __init__(self, lo: int = None, hi: int = None):
        """
        初始化

        Args:
            lo / hi: 区块窗口
        Returns:
            None
        """
        self.lo = lo or config.MODEL_WINDOW_START_BLOCK
        self.hi = hi or config.MODEL_WINDOW_END_BLOCK
        self.client = SQDClient()
        self.pool = RpcPool('base')
        self.hot_wallets = {}
        self.stats = {}

    # ============================================================
    #  通用工具
    # ============================================================
    @staticmethod
    def _is_addr(a) -> bool:
        """合法 EVM 地址判定。"""
        a = str(a or '').lower()
        return a.startswith('0x') and len(a) == 42 and int(a, 16) != 0

    def _existing_labels(self) -> set:
        """已在 agent / bot 组的地址（清洗逻辑 3：避免标签冲突）。"""
        out = set()
        for g in ('agent', 'bot'):
            for r in read_address_csv(config.ADDR_DIR / config.GROUPS[g][0]):
                out.add(r['address'].lower())
        return out

    # ============================================================
    #  阶段1：反推 CEX 热钱包
    # ============================================================
    def _find_hot_wallets(self, lo: int, hi: int) -> dict:
        """
        按「散播型转账」行为反推 CEX 热钱包（清洗逻辑 1）

        ★ 判据: 单地址向大量不同接收方转账，且接收方重复率低。
           这精准刻画「交易所给用户提币」的形态，不依赖任何标签表。

        Args:
            lo / hi: 区块范围
        Returns:
            dict: {hot_wallet: n_recipients}
        """
        log('阶段1：按散播行为反推 CEX 热钱包', 'step')
        q = {
            'type': 'evm', 'fromBlock': lo, 'toBlock': hi,
            'fields': {'block': {'number': True},
                       'transaction': {'from': True, 'to': True, 'value': True}},
            'transactions': [{}],
        }
        senders = defaultdict(lambda: {'recips': set(), 'n': 0})
        n_tx = 0
        for blk in self.client.stream(q):
            for tx in blk.get('transactions', []):
                f = (tx.get('from') or '').lower()
                t = (tx.get('to') or '').lower()
                if not self._is_addr(f) or not self._is_addr(t):
                    continue
                n_tx += 1
                s = senders[f]
                s['recips'].add(t)
                s['n'] += 1

        hot = {}
        for addr, s in senders.items():
            nr = len(s['recips'])
            if nr < self.MIN_RECIPIENTS:
                continue
            if nr / max(s['n'], 1) < self.MIN_UNIQUE_RATIO:
                continue                 # 接收方重复率高 = 跟固定合约交互，不是散播
            hot[addr] = nr

        hot = dict(sorted(hot.items(), key=lambda x: -x[1])[:self.MAX_HOT_WALLETS])
        log(f'  扫描 {n_tx:,} 笔交易 → {len(senders):,} 个发送方 → '
            f'{len(hot)} 个散播型热钱包', 'ok')
        for a, n in list(hot.items())[:5]:
            print(f'    {a}  → {n:,} 个不同接收方')
        self.hot_wallets = hot
        self.stats['扫描交易数'] = f'{n_tx:,}'
        self.stats['热钱包数'] = len(hot)
        return hot

    # ============================================================
    #  阶段2：取接收方
    # ============================================================
    def _collect_recipients(self, hot: dict, lo: int, hi: int) -> dict:
        """
        取热钱包的转账接收方 = 人类样本候选

        Args:
            hot:     阶段1 的热钱包
            lo / hi: 区块范围
        Returns:
            dict: {recipient: {n, cexs}}
        """
        log('阶段2：取热钱包的提币接收方', 'step')
        q = {
            'type': 'evm', 'fromBlock': lo, 'toBlock': hi,
            'fields': {'block': {'number': True},
                       'transaction': {'from': True, 'to': True}},
            'transactions': [{'from': list(hot)}],
        }
        recips = defaultdict(lambda: {'n': 0, 'cexs': set()})
        for blk in self.client.stream(q):
            for tx in blk.get('transactions', []):
                f = (tx.get('from') or '').lower()
                t = (tx.get('to') or '').lower()
                if f not in hot or not self._is_addr(t):
                    continue
                recips[t]['n'] += 1
                recips[t]['cexs'].add(f[:10])
        log(f'  得到 {len(recips):,} 个接收方', 'ok')
        return recips

    def _filter_active(self, addrs: list, lo: int, hi: int) -> dict:
        """
        阶段3：活跃度筛选 —— 🔴 2026-09-03 实测新增

        【为什么必须有这一步】
          实测: 未经活跃度筛选的 CEX 提现收款方，
                12,000 块窗口 → 2.67 条活动/地址
               144,000 块窗口 → 2.64 条活动/地址（12 倍窗口只多 8% 活动）
          ⇒ 绝大多数收款方是「一次性提币地址」: 提完就休眠，永远凑不满
            MIN_TX_FOR_FEATURES(20)，整组在特征阶段被全部剔除。
          必须在候选池里筛出「真的在用这个钱包」的那一小撮。

        【判据】 探测窗口内自发交易数落在 [MIN_ACTIVE_TX, MAX_ACTIVE_TX]。
                下限剔休眠地址，上限剔混进来的 bot（实测见过 184,394 笔/33h）。
                只看「用了多少次」这一个标量，不看什么时候用、用得多规律 ——
                🔴 这是刻意的：任何涉及时间分布的筛选都会让 f02 的
                   entropy_24h / night_ratio 变成「用答案筛样本」。
                ⚠️ 代价: 活动量本身成了弱选择条件，所以 n_acts / acts_per_day
                   已列入 SCALE_FEATURES 排除在模型之外，不得作为特征使用。

        Args:
            addrs:   候选地址（阶段2 产出，已排除 agent/bot 标签）
            lo / hi: 探测窗口
        Returns:
            dict: {address: 该窗口内自发交易数}，已过滤掉不活跃的
        """
        log(f'阶段3：活跃度筛选（{len(addrs):,} 个候选，'
            f'窗口 {hi-lo:,} 块）', 'step')
        act = defaultdict(int)
        # SQD 单次可过滤 1000 个地址，候选通常一批就够
        for i in range(0, len(addrs), config.SQD_MAX_ADDR_PER_QUERY):
            chunk = addrs[i:i + config.SQD_MAX_ADDR_PER_QUERY]
            q = {
                'type': 'evm', 'fromBlock': lo, 'toBlock': hi,
                'fields': {'block': {'number': True},
                           'transaction': {'from': True}},
                'transactions': [{'from': chunk}],
            }
            for blk in self.client.stream(q):
                for tx in blk.get('transactions', []):
                    f = (tx.get('from') or '').lower()
                    if f:
                        act[f] += 1

        keep = {a: n for a, n in act.items()
                if self.MIN_ACTIVE_TX <= n <= self.MAX_ACTIVE_TX}
        n_hi = sum(1 for n in act.values() if n > self.MAX_ACTIVE_TX)
        log(f'  {len(act):,} 个有活动，其中 {len(keep):,} 个落在 '
            f'[{self.MIN_ACTIVE_TX}, {self.MAX_ACTIVE_TX}] 笔的人类合理区间', 'ok')
        if n_hi:
            log(f'  另剔除 {n_hi} 个活动数 > {self.MAX_ACTIVE_TX} 的高频地址'
                f'（疑似混入的 bot/热钱包 EOA）', 'warn')
        if act:
            import numpy as _np
            v = _np.array(list(act.values()))
            log(f'  候选活动数分布: 中位 {_np.median(v):.0f} / '
                f'p90 {_np.percentile(v, 90):.0f} / max {v.max()}', 'info')
        return keep

    def _filter_eoa(self, addrs: list, limit: int) -> list:
        """
        只保留 EOA（清洗逻辑 2：合约不是人）

        Args:
            addrs: 候选地址
            limit: 最多保留多少个（RPC 有限速，不做全量）
        Returns:
            list: EOA 地址
        """
        log(f'过滤合约（只保留 EOA），最多检查 {limit*2} 个...', 'step')
        out = []
        for a in addrs[: limit * 2]:
            try:
                if len(self.pool.call('eth_getCode', [a, 'latest'])) <= 2:
                    out.append(a)
            except Exception:
                continue
            if len(out) >= limit:
                break
        log(f'  {len(out)} 个是 EOA', 'ok')
        return out

    # ============================================================
    #  ① 主流程
    # ============================================================
    def fetch_humans(self, blocks: int = 20000, target: int = 300,
                     save_to_csv: bool = True) -> pd.DataFrame:
        """
        ① 采集人类样本候选（Base 链上反推 CEX 提币接收方）

        【数据源】 SQD stream 两阶段扫描
                 阶段1: 全链交易 → 按 from 聚合 → 散播型地址 = CEX 热钱包
                 阶段2: 这些热钱包的接收方 = 人类样本候选
                 - 无需 API Key，SQD 公开 Portal 免费
                 - 不依赖 Dune labels / Coinglass（后者只有 Ethereum）

        【指标含义】 一句话: "从交易所提币的地址，大概率是自然人"
                    交易所提币必然经过 KYC ⇒ 对「这个地址背后是人」提供弱先验。

        【返回字段】
                  · address     — 提币接收方（★人类样本候选）
                  · source      — base_cex_withdrawal
                  · note        — n=N cex=前缀列表
                  · n_from_cex  — 收到几次提币

        【阈值解读】
                    · 热钱包判据: n_recipients >= 40 且 唯一接收方占比 >= 0.8
                    · 接收方必须是 EOA（合约不是人）

        【如何使用】
                   1) 🔴 **反循环设计**: 本模块**绝不用昼夜节律筛选** ——
                      用节律筛人类、又用 entropy_24h 当特征 = 用答案筛样本。
                      只用「收到 CEX 提币 + 是 EOA」两条与节律无关的判据，
                      ⇒ entropy_24h / night_ratio 因此是独立证据，可放心进模型
                   2) 🔴 论文必须交代: CEX 提现只是弱先验不是证明，
                      机构/做市商/bot 都可能提币，必须做 5-30% 标签噪声敏感性分析
                   3) 建议再与 ENS/Farcaster 绑定交叉，至少命中 2 条信号

        Args:
            blocks:      从窗口末尾往回扫多少区块
            target:      目标样本数
            save_to_csv: 是否写入 data/addresses/human_cex.csv
        Returns:
            pd.DataFrame
        """
        hi = self.hi
        lo = max(self.lo, hi - blocks)
        log(f'区块 {lo:,} → {hi:,}（{hi-lo:,} 块）', 'info')

        hot = self._find_hot_wallets(lo, hi)
        if not hot:
            log('未发现散播型热钱包，试试加大 --blocks 或降低 MIN_RECIPIENTS', 'warn')
            return pd.DataFrame()

        recips = self._collect_recipients(hot, lo, hi)
        if not recips:
            return pd.DataFrame()

        # 3. 排除已有标签的地址
        exclude = self._existing_labels()
        cand = [a for a in sorted(recips, key=lambda x: -recips[x]['n'])
                if a not in exclude]
        log(f'  排除已在 agent/bot 组的地址后剩 {len(cand):,} 个', 'info')

        # 4. 🔴 活跃度筛选 —— 必须在 EOA 之前：
        #    批量扫描一次筛掉 90%+ 的休眠地址，剩下的才值得逐个花 RPC 调用验 EOA
        act = self._filter_active(cand, max(self.lo, hi - self.ACTIVE_PROBE_BLOCKS), hi)
        if not act:
            log('没有候选达到活跃门槛 —— 调大 --blocks 或降低 MIN_ACTIVE_TX', 'warn')
            return pd.DataFrame()
        cand = sorted(act, key=lambda x: -act[x])

        # 5. 只保留 EOA
        eoas = self._filter_eoa(cand, target)
        if not eoas:
            return pd.DataFrame()

        rows = [{'address': a, 'source': 'base_cex_withdrawal',
                 'note': f"n={recips[a]['n']} act={act[a]} "
                         f"cex={'|'.join(sorted(recips[a]['cexs'])[:2])}",
                 'n_from_cex': recips[a]['n'], 'n_active': act[a]} for a in eoas]
        df = pd.DataFrame(rows)
        self.stats['人类样本候选'] = len(df)

        if save_to_csv:
            out = config.ADDR_DIR / config.GROUPS['human'][0]
            with out.open('w', newline='', encoding='utf-8') as f:
                f.write('# 人类样本候选 —— 由 HumanDataSource 在 Base 链上反推\n')
                f.write(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write('# 判据：收到「散播型热钱包」转账 + 探测窗口内自发交易 '
                        f'>= {self.MIN_ACTIVE_TX} 笔 + 是 EOA\n')
                f.write('# 🔴 活跃度筛选只看「有没有在用」，不看时间分布 ——\n')
                f.write('#    任何按时段筛选都会让 entropy_24h / night_ratio 循环失效\n')
                f.write('# 🔴 反循环：绝不用昼夜节律筛选，'
                        '否则 entropy_24h 特征会变成用答案筛样本\n')
                f.write('# 🔴 CEX 提现只是弱先验，必须做 5-30% 标签噪声敏感性分析\n')
                f.write('address,source,note\n')
                for r in rows:
                    f.write(f"{r['address']},{r['source']},"
                            f"{str(r['note']).replace(',', ';')}\n")
            log(f'写入 {out.relative_to(config.ROOT)}（{len(df)} 个）', 'ok')
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

    def validate_humans(self, df: pd.DataFrame,
                        context: str = '人类样本') -> bool:
        """
        校验 ① 的产出

        检查项: 非空 / 列齐全 / 地址合法且无重复 /
               与 agent·bot 组无交集 / ⚠️样本量 / 🔴弱先验提示

        Args:
            df / context
        Returns:
            bool
        """
        import re
        issues, warns = [], []
        addr_re = re.compile(r'^0x[0-9a-f]{40}$')

        if df is None or df.empty:
            issues.append('结果为空')
            return self._emit(context, issues, warns)
        miss = [c for c in ('address', 'source', 'note') if c not in df.columns]
        if miss:
            issues.append(f'缺列: {miss}')
            return self._emit(context, issues, warns)

        bad = df[~df['address'].astype(str).str.match(addr_re)]
        if len(bad):
            issues.append(f'{len(bad)} 个地址格式非法')
        dup = int(df['address'].duplicated().sum())
        if dup:
            issues.append(f'{dup} 个地址重复')

        overlap = set(df['address']) & self._existing_labels()
        if overlap:
            issues.append(f'{len(overlap)} 个地址与 agent/bot 组冲突')

        if len(df) < 100:
            warns.append(f'仅 {len(df)} 个样本，建议加大 --blocks 或 --target')

        warns.append('🔴 提醒：CEX 提现仅为弱先验，'
                     '必须做 5-30% 标签噪声敏感性分析')
        warns.append('✅ 反循环已保证：未用昼夜节律筛选，'
                     'entropy_24h/night_ratio 可放心进模型')
        return self._emit(context, issues, warns)


# ================================================================
#  HumanFetcher — 适配 BaseFetcher
# ================================================================
class HumanFetcher(BaseFetcher):
    """薄适配层：把 HumanDataSource 接进项目的 Fetcher 体系。"""

    name = 'human'
    desc = '人类样本候选（Base 链上反推 CEX 提币）'
    output = ''
    quality = '★★ 弱先验'
    needs_key = False

    def __init__(self, blocks=20000, target=300, **kw):
        super().__init__(**kw)
        self.blocks = int(blocks or 20000)
        self.target = int(target or 300)
        self.ds = HumanDataSource()
        self.df = pd.DataFrame()

    def fetch(self) -> list:
        self.df = self.ds.fetch_humans(blocks=self.blocks, target=self.target,
                                       save_to_csv=True)
        self.ds.validate_humans(self.df)
        self.stats.update(self.ds.stats)
        return []

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--blocks', type=int, default=20000,
                        help='从窗口末尾往回扫多少区块')
        ap.add_argument('--target', type=int, default=300, help='目标样本数')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(description='人类样本候选采集（Base 链上反推）')
    HumanFetcher.add_args(ap)
    args = ap.parse_args()
    HumanFetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
