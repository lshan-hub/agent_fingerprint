#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_mev_bots.py
@Description:
    MEV Bot 识别（SQD 日志扫描）—— 传统脚本 bot 负样本，附 validate_* 校验。

    ============================================================
    一、 认证与限频
    ============================================================
      方式:   SQD Network 公开 Portal，无需 API Key、无需注册、零成本
      Base:   https://portal.sqd.dev/datasets/base-mainnet/stream
      限频:   无硬限流，但单次响应只推进约 396 区块（SQDClient.stream 已自动分页）

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — 原子套利地址 (Atomic Arbitrage) ★bot 负样本 ──────────────────┐
    │  接口:  SQD stream，过滤全链三类 DEX Swap 事件的 topic0                   │
    │           Uniswap V2 Swap(address,uint256,uint256,uint256,uint256,address)│
    │           Uniswap V3 Swap(address,address,int256,int256,uint160,...)      │
    │           Uniswap V4 Swap(bytes32,address,int128,int128,uint160,...)      │
    │  函数:  fetch_bots() (内部 _build_query / _scan)                          │
    │  覆盖:  Base 全链（不限特定合约，覆盖绝大多数交易量）                      │
    │  更新频率: 实时（随链上区块推进）                                          │  ★必填
    │  更新时间: 无固定刷新点                                                    │  ★必填
    │  API KEY: 否 —— SQD 公开 Portal，完全免费                                 │  ★必填
    │  历史范围: ⭐ 全历史；默认从链头往回扫 --blocks 个区块                     │  ★必填
    │  产出: data/sources/mev_bots.csv                                          │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ ★ 判据: 同一笔交易内出现 >= N 次 DEX Swap = 原子套利 = 必然是脚本

      ※ 🎯 为什么这个判据可靠:
        · 自包含 —— 不依赖 Dune labels / Etherscan 标签（标签表覆盖不全且会改名）
        · 假阳性极低 —— 人类不可能在一笔交易里手工串起多次 swap
        · 直接对应「传统脚本 bot」的定义（预编译逻辑、抢区块）

      ※ ⚠️ 已知采样偏差: 原子套利只是 MEV 的一种，这样选出的 bot 偏向套利型，
        缺少清算 bot、抢跑 bot、狙击 bot。
        🔴 论文里必须说明 bot 类别构成，否则「bot」这一类的定义会被质疑。

      链上原始返回        → 统一字段     → 说明
      [单笔交易内 swap 数] → max_swaps    该地址单笔最多串了几次 swap
      tx.from            → address      套利交易的发起人（★bot 负样本）
      [达标交易计数]      → arb_tx       原子套利交易笔数
      [去重区块数]        → n_blocks     活跃区块数（排除一次性脚本）

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. 单笔交易内 swap 次数 < min_swaps_in_tx ⇒ 不算原子套利，跳过
      2. 取不到 tx.from ⇒ 跳过
      3. 提纯门槛:
         · arb_tx   >= min_arb     —— 偶发的多跳 swap 不算 bot
         · n_blocks >= min_blocks  —— 排除一次性脚本

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🥇 三组真值中「传统脚本 bot」负样本的**唯一来源**，真值质量 ★★★。
      ⚠️ 这一类是三分类任务的**主要混淆来源**：AI agent 做自主交易时，
        链上表现可能与 MEV bot 高度相似。混淆矩阵的 bot↔agent 那一格
        必须单独分析，并在论文中讨论。
'''

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils._base import BaseFetcher, log, topic0, write_csv  # noqa: E402
from src.utils.sqd_client import SQDClient  # noqa: E402


# ================================================================
#  MevBotDataSource — 原子套利识别
# ================================================================
class MevBotDataSource(object):
    """
    MEV Bot 数据采集与管理（用行为定义反推，不依赖外部标签表）

    核心函数 (共 1 个):
      ① fetch_bots(blocks, from_block, min_swaps_in_tx, min_arb, min_blocks)
          扫全链 DEX Swap 事件 → 单笔多 swap = 原子套利 = 脚本 bot

    辅助函数:
      - _build_query()  构造 SQD 查询体
      - _scan()         扫描并按发起人聚合

    校验函数: validate_bots

    Args:
        client: 可选的 SQDClient 实例；缺省时自建
    """

    # 主流 DEX 的 Swap 事件（覆盖 Base 上绝大多数交易量）
    SWAP_TOPICS = {
        'uniswap_v2': topic0('Swap(address,uint256,uint256,uint256,uint256,address)'),
        'uniswap_v3': topic0('Swap(address,address,int256,int256,uint160,uint128,int24)'),
        'uniswap_v4': topic0(
            'Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)'),
    }

    # 单地址 max_swaps 超过这个值属于异常复杂的套利路径，校验器会提示
    SANITY_MAX_SWAPS = 30

    def __init__(self, client: SQDClient = None):
        """
        初始化

        Args:
            client: 可选的 SQDClient；缺省自建
        Returns:
            None
        """
        self.client = client or SQDClient()
        self.n_swap = 0
        self.n_arb_tx = 0
        self.raw_stats = {}

    # ============================================================
    #  通用工具
    # ============================================================
    @staticmethod
    def _safe_int(v, default: int = 0):
        """安全转 int。"""
        try:
            return default if v is None or v == '' else int(v)
        except (TypeError, ValueError):
            return default

    def _build_query(self, lo: int, hi: int) -> dict:
        """
        构造 SQD 查询体

        Args:
            lo / hi: 区块范围
        Returns:
            dict
        """
        return {
            'type': 'evm',
            'fromBlock': lo,
            'toBlock': hi,
            'fields': {
                'block': {'number': True, 'timestamp': True},
                'log': {'topics': True, 'transactionIndex': True},
                'transaction': {'from': True, 'transactionIndex': True},
            },
            'logs': [{'topic0': list(self.SWAP_TOPICS.values()), 'transaction': True}],
        }

    def _scan(self, lo: int, hi: int, min_swaps_in_tx: int) -> dict:
        """
        扫全链 Swap 事件，找出「单笔交易内 swap 次数 >= 阈值」的发起人

        Args:
            lo / hi:          区块范围
            min_swaps_in_tx:  原子套利判据阈值
        Returns:
            dict: {address: {arb_tx, max_swaps, swaps, blocks}}
        """
        stats = defaultdict(lambda: {'arb_tx': 0, 'max_swaps': 0,
                                     'swaps': 0, 'blocks': set()})
        n_swap = n_tx_arb = 0

        for blk in self.client.stream(self._build_query(lo, hi)):
            bn = blk.get('header', {}).get('number')
            tx_from = {t.get('transactionIndex'): (t.get('from') or '').lower()
                       for t in blk.get('transactions', [])}

            per_tx = defaultdict(int)
            for lg in blk.get('logs', []):
                per_tx[lg.get('transactionIndex')] += 1
                n_swap += 1

            for txi, cnt in per_tx.items():
                if cnt < min_swaps_in_tx:      # 1. 不达标不算原子套利
                    continue
                sender = tx_from.get(txi)
                if not sender:                 # 2. 取不到发起人
                    continue
                n_tx_arb += 1
                s = stats[sender]
                s['arb_tx'] += 1
                s['swaps'] += cnt
                s['max_swaps'] = max(s['max_swaps'], cnt)
                s['blocks'].add(bn)

        self.n_swap, self.n_arb_tx = n_swap, n_tx_arb
        log(f'扫描完成：{n_swap:,} 条 Swap 事件，'
            f'{n_tx_arb:,} 笔交易达到 >={min_swaps_in_tx} 次 swap 的原子套利判据', 'ok')
        return stats

    # ============================================================
    #  ① 原子套利 → 传统脚本 bot 负样本
    # ============================================================
    def fetch_bots(self, blocks: int = 5000, from_block: int = 0,
                   min_swaps_in_tx: int = 2, min_arb: int = 5,
                   min_blocks: int = 3,
                   save_to_csv: bool = True) -> pd.DataFrame:
        """
        ① 用原子套利判据识别 MEV bot（传统脚本 bot 负样本）

        【数据源】 SQD stream，过滤全链三类 DEX Swap 事件的 topic0
                 - Uniswap V2 / V3 / V4 的 Swap 事件
                 - 无需 API Key，SQD 公开 Portal 免费
                 - 不限特定合约，覆盖 Base 上绝大多数交易量

        【指标含义】 一句话: "同一笔交易内串起多次 swap = 必然是脚本"
                    原子套利要求在单笔交易内完成「低买 + 高卖」，
                    人类不可能手工做到 —— 这是最干净的 bot 行为定义。
                    🎯 判据自包含，不依赖 Dune labels / Etherscan 标签
                       （标签表覆盖不全且会改名）。

        【返回字段】
                  · address    — 套利交易发起人（★bot 负样本）
                  · source     — mev_atomic_arb
                  · note       — arb_tx=N max_swaps=N blocks=N
                  · arb_tx     — 原子套利交易笔数
                  · max_swaps  — 单笔最多串了几次 swap
                  · n_blocks   — 活跃区块数

        【阈值解读】
                    · min_swaps_in_tx=2  → 原子套利的最低定义
                    · arb_tx < 5         → 偶发多跳 swap，可能是普通用户，剔除
                    · n_blocks < 3       → 一次性脚本，剔除
                    · max_swaps > 30     → 异常复杂的套利路径，值得人工核查

        【如何使用】
                   1) 直接作为三分类的 bot 负样本（真值质量 ★★★）
                   2) ⚠️ 采样偏向套利型，缺清算/抢跑/狙击 bot ——
                      论文里必须说明 bot 类别构成，否则定义会被质疑
                   3) ⚠️ 这一类是三分类的主要混淆来源：AI agent 做自主交易时
                      链上表现可能与 MEV bot 高度相似。混淆矩阵的 bot↔agent
                      那一格必须单独分析
                   4) 与 agent 样本取交集时，重叠地址需人工裁决（不能既是 agent 又是 bot）

        Args:
            blocks:          从链头往回扫多少区块（Base 2s/块）
            from_block:      指定起始区块（覆盖 blocks）
            min_swaps_in_tx: 单笔交易内至少几次 swap 才算原子套利
            min_arb:         单地址至少多少笔套利交易才收录
            min_blocks:      至少活跃多少个不同区块
            save_to_csv:     是否写入 data/sources/mev_bots.csv
        Returns:
            pd.DataFrame
        """
        head = self.client.head()
        hi = head
        lo = from_block or max(0, head - blocks)

        log('① 扫 Base 全链 DEX Swap 事件（识别原子套利）', 'step')
        log(f'  区块 {lo:,} → {hi:,}（{hi-lo:,} 块）')
        log(f'  判据：单笔交易内 >= {min_swaps_in_tx} 次 swap')

        raw = self._scan(lo, hi, min_swaps_in_tx)
        self.raw_stats = raw
        if not raw:
            log('未发现原子套利。试试加大 blocks', 'warn')
            return pd.DataFrame()

        rows = []
        for addr, s in raw.items():
            if s['arb_tx'] < min_arb or len(s['blocks']) < min_blocks:   # 3. 提纯
                continue
            rows.append({
                'address': addr, 'source': 'mev_atomic_arb',
                'note': (f"arb_tx={s['arb_tx']} max_swaps={s['max_swaps']} "
                         f"blocks={len(s['blocks'])}"),
                'arb_tx': s['arb_tx'], 'max_swaps': s['max_swaps'],
                'n_blocks': len(s['blocks']),
            })
        if not rows:
            log(f'过滤后无剩余（门槛 arb_tx>={min_arb} blocks>={min_blocks}）', 'warn')
            return pd.DataFrame()

        df = (pd.DataFrame(rows)
              .sort_values('arb_tx', ascending=False)
              .reset_index(drop=True))
        log(f'  候选 {len(raw):,} 个 → 过滤后 {len(df):,} 个 MEV bot', 'ok')

        if save_to_csv:
            write_csv('mev_bots.csv', df.to_dict('records'), list(df.columns),
                      note=('MEV bot（传统脚本 bot 负样本，真值质量 ★★★）\n'
                            '判据：单笔交易内多次 DEX Swap = 原子套利 = 必然是脚本\n'
                            '⚠️ 采样偏向套利型，缺清算/抢跑/狙击 bot，'
                            '论文需说明类别构成\n'
                            f'区块 {lo}-{hi} | 门槛 arb_tx>={min_arb} '
                            f'blocks>={min_blocks}'))
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

    def validate_bots(self, df: pd.DataFrame,
                      context: str = 'MEV bot') -> bool:
        """
        校验 ① 的产出

        检查项:
          1. 非空 / 必需列齐全
          2. 地址格式合法且无重复
          3. arb_tx / max_swaps / n_blocks 取值合理
          4. ⚠️ max_swaps 异常高（可能是复杂套利路径，值得核查）
          5. ⚠️ 样本量是否够训练
          6. 🔴 强制提示: 采样偏差 + bot↔agent 混淆

        Args:
            df / context
        Returns:
            bool
        """
        import re
        issues, warns = [], []
        addr_re = re.compile(r'^0x[0-9a-f]{40}$')
        need = ['address', 'source', 'note', 'arb_tx', 'max_swaps', 'n_blocks']

        if df is None or df.empty:
            issues.append('结果为空')
            return self._emit(context, issues, warns)
        miss = [c for c in need if c not in df.columns]
        if miss:
            issues.append(f'缺列: {miss}')
            return self._emit(context, issues, warns)

        bad = df[~df['address'].astype(str).str.match(addr_re)]
        if len(bad):
            issues.append(f'{len(bad)} 个地址格式非法')
        dup = int(df['address'].duplicated().sum())
        if dup:
            issues.append(f'{dup} 个地址重复（聚合逻辑有误）')
        if (df['arb_tx'] < 1).any():
            issues.append('存在套利笔数 < 1 的行')
        if (df['max_swaps'] < 2).any():
            issues.append('存在 max_swaps < 2 的行（不符合原子套利定义）')
        if (df['n_blocks'] > df['arb_tx']).any():
            issues.append('存在活跃区块数 > 套利笔数的行（统计逻辑有误）')

        n_odd = int((df['max_swaps'] > self.SANITY_MAX_SWAPS).sum())
        if n_odd:
            warns.append(f'{n_odd} 个地址 max_swaps > {self.SANITY_MAX_SWAPS}，'
                         f'属异常复杂的套利路径，建议人工核查')

        if len(df) < 100:
            warns.append(f'仅 {len(df)} 个地址，样本偏少，建议加大 blocks')

        warns.append('🔴 提醒：采样偏向套利型，缺清算/抢跑/狙击 bot，'
                     '论文需说明 bot 类别构成')
        warns.append('🔴 提醒：bot↔agent 是三分类的主要混淆来源，'
                     '混淆矩阵那一格需单独分析')
        return self._emit(context, issues, warns)


# ================================================================
#  MevBotFetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class MevBotFetcher(BaseFetcher):
    """
    薄适配层：把 MevBotDataSource 接进项目的 Fetcher 体系。

    数据逻辑全在 MevBotDataSource；本类只负责参数透传与产出对接。
    """

    name = 'mev'
    desc = 'MEV bot（传统脚本 bot 负样本）'
    output = 'mev_bots.csv'
    quality = '★★★'
    needs_key = False
    fields = ['address', 'source', 'note', 'arb_tx', 'max_swaps', 'n_blocks']

    def __init__(self, blocks=5000, from_block=0, min_swaps_in_tx=2,
                 min_arb=5, min_blocks=3, **kw):
        super().__init__(**kw)
        self.blocks = int(blocks or 5000)
        self.from_block = int(from_block or 0)
        self.min_swaps_in_tx = int(min_swaps_in_tx or 2)
        self.min_arb = int(min_arb or 5)
        self.min_blocks = int(min_blocks or 3)
        self.ds = MevBotDataSource()
        self.df = pd.DataFrame()

    def fetch(self) -> list:
        self.df = self.ds.fetch_bots(
            blocks=self.blocks, from_block=self.from_block,
            min_swaps_in_tx=self.min_swaps_in_tx,
            min_arb=self.min_arb, min_blocks=self.min_blocks,
            save_to_csv=True)
        self.ds.validate_bots(self.df)
        self.stats['Swap 事件'] = f'{self.ds.n_swap:,}'
        self.stats['原子套利交易'] = f'{self.ds.n_arb_tx:,}'
        self.stats['bot（过滤后）'] = len(self.df)
        return []      # 已在 fetch_bots 内落盘

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--blocks', type=int, default=5000,
                        help='从链头往回扫多少区块（Base 2s/块）')
        ap.add_argument('--from-block', type=int, default=0)
        ap.add_argument('--min-swaps-in-tx', type=int, default=2,
                        help='单笔交易内至少几次 swap 才算原子套利')
        ap.add_argument('--min-arb', type=int, default=5,
                        help='单地址至少多少笔套利交易才收录（提纯）')
        ap.add_argument('--min-blocks', type=int, default=3,
                        help='至少活跃多少个不同区块（排除一次性脚本）')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='MEV bot 识别（原子套利判据，传统脚本 bot 负样本）')
    MevBotFetcher.add_args(ap)
    args = ap.parse_args()
    MevBotFetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
