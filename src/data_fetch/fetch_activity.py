#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_activity.py
@Description:
    统一活动流采集 —— 三组真值地址的链上活动，特征计算的直接输入。

    ============================================================
    一、 为什么需要「统一活动流」而不是「交易史」
    ============================================================
    🔴 2026-09-03 实测发现，三组的链上足迹形态**完全不同**:

        组             是合约   nonce(自己发出的交易数)   6000 块内 tx.from
        Olas Safe      ✅ 是    1（只有部署那一笔）       0
        x402 付款方     否      2 ~ 1,079                 1
        MEV bot        否      749,745 ~ 2,911,255       5,766

      · Olas agent 是 **Gnosis Safe 合约账户** —— 合约不能主动发交易，
        它的活动体现在「被 owner/relayer 调用」，即出现在 tx.to
      · x402 付款方走 **EIP-3009 元交易** —— Facilitator 代付 gas 并提交，
        付款方自己几乎不发交易；其活动 100% 体现在 AuthorizationUsed 事件
        （实测 8 个地址在 6000 块内有 7,591 条授权事件）

      ⇒ 用 tx.from 采集，Olas 和 x402 几乎采不到任何数据，课题直接做不下去。

    ★ 正确口径: Δt = 该地址相邻两次【链上活动】的间隔
        ① tx_from  作为交易发起人      （MEV bot / 人类 的主要形态）
        ② tx_to    作为交易接收方被调用（Olas Safe 的主要形态）
        ③ event    出现在事件参数中    （x402 付款方的主要形态）

      这个口径反而更正确 —— 它捕捉的是「**决策频率**」而非「交易发起方式」，
      三组因此可比。🔴 论文中必须说明此口径及其理由。

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — 统一活动流 (Unified Activity Stream) ★特征输入 ──────────────┐
    │  接口:  SQD stream，三类过滤器并行                                        │
    │           transactions:[{from:[...]}]  → tx_from                          │
    │           transactions:[{to:[...]}]    → tx_to                            │
    │           logs:[{topic0:[AuthorizationUsed / Transfer]}] → event          │
    │  函数:  fetch_group(group) / fetch_all_groups()                           │
    │  更新频率: 实时（随链上区块推进）                                          │  ★必填
    │  API KEY: 否 —— SQD 公开 Portal，完全免费                                 │  ★必填
    │  历史范围: config.MODEL_WINDOW_START_BLOCK ~ END_BLOCK                    │  ★必填
    │            = 2026-03-01 ~ 2026-09-03，8,064,661 区块 ≈ 187 天            │
    │  产出: data/raw/{chain}/{group}/{address}.json                            │
    └───────────────────────────────────────────────────────────────────────────┘

      统一后的活动记录字段（与 compute_features 的输入契约一致）:
        字段            来源                              说明
        timeStamp      block.timestamp                   ★ Δt 的唯一依据
        blockNumber    block.number                      同区块判定
        txIndex        transaction.transactionIndex      块内排序
        kind           tx_from / tx_to / event           ★ 活动类型（新增）
        to             transaction.to                    交互对手
        gasPrice       transaction.gasPrice              仅 tx_from 有意义
        nonce          transaction.nonce                 仅 tx_from 有意义
        value          transaction.value
        isError        由 status 转换（0=成功，与 Etherscan 同口径）
        methodId       transaction.sighash

      ⚠️ tx_to / event 类活动没有 gasPrice / nonce（不是该地址付的 gas），
        统一填 '0'，由 compute_features 的 f_gas / f_nonce 自行处理缺失。
        这是**已知的特征可得性差异**，论文必须交代。

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. 🔴 每地址活动封顶 config.MAX_ACTIVITY_PER_ADDRESS(3000) 条 ——
         不封顶单个 MEV bot 6 个月约 525 MB，700 个就是 359 GB，远超 5 GB 预算。
         行为特征只需几千条就能算出稳定统计量，封顶不损失信息。
         封顶策略: 从窗口**末尾往前**取（保留最新行为）
      2. 按 (timeStamp, txIndex) 去重排序
      3. 地址批量过滤，每批 config.SQD_MAX_ADDR_PER_QUERY(1000) 个
      4. 结果落盘缓存，重跑不重复消耗

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🔗 **管线枢纽** —— 上承 data/addresses/ 的三组真值清单，
        下接 src/feature/ 的特征计算。
        取代 fetch_txs.py 成为默认采集器（后者只按 tx.from 采，
        对 Olas/x402 会漏采，仅在需要 Etherscan 后端时使用）。
'''

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils._base import BaseFetcher, log, topic0  # noqa: E402
from src.utils.common import read_address_csv, save_cache  # noqa: E402
from src.utils.sqd_client import SQDClient  # noqa: E402


# ================================================================
#  ActivityDataSource — 统一活动流采集
# ================================================================
class ActivityDataSource(object):
    """
    三组真值地址的统一活动流采集与管理

    核心函数 (共 2 个):
      ① fetch_group(group)      单组地址的统一活动流
      ② fetch_all_groups()      三组全采 + 汇总统计

    辅助函数:
      - _scan_tx(addrs, key)    扫 tx.from / tx.to
      - _scan_events(addrs)     扫事件参数中出现该地址的记录
      - _merge_and_cap()        合并去重 + 按地址封顶

    校验函数: validate_activity

    Args:
        chain:      目标链名；缺省 config.PRIMARY_CHAIN
        lo / hi:    区块窗口；缺省 config.MODEL_WINDOW_*
        cap:        每地址活动上限；缺省 config.MAX_ACTIVITY_PER_ADDRESS
    """

    USDC_BASE = '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913'
    TOPIC_AUTH = topic0('AuthorizationUsed(address,bytes32)')
    TOPIC_TRANSFER = topic0('Transfer(address,address,uint256)')

    def __init__(self, chain: str = None, lo: int = None, hi: int = None,
                 cap: int = None):
        """
        初始化

        Args:
            chain / lo / hi / cap: 见类文档
        Returns:
            None
        """
        self.chain = chain or config.PRIMARY_CHAIN
        self.lo = lo or config.MODEL_WINDOW_START_BLOCK
        self.hi = hi or config.MODEL_WINDOW_END_BLOCK
        self.cap = cap or config.MAX_ACTIVITY_PER_ADDRESS
        self.client = SQDClient()
        self.stats = defaultdict(int)

    # ============================================================
    #  通用工具
    # ============================================================
    @staticmethod
    def _to_dec(v) -> str:
        """SQD 返回十六进制，统一转十进制字符串（与 Etherscan 同口径）。"""
        if v is None:
            return '0'
        if isinstance(v, str) and v.startswith('0x'):
            try:
                return str(int(v, 16))
            except ValueError:
                return '0'
        return str(v)

    @staticmethod
    def _safe_int(v, default: int = 0):
        """安全转 int（支持 0x 前缀）。"""
        try:
            if v is None or v == '':
                return default
            return int(v, 16) if isinstance(v, str) and v.startswith('0x') else int(v)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _topic_to_addr(t: str) -> str:
        """32 字节 topic → 20 字节地址（小写）。"""
        return ('0x' + t[-40:]).lower()

    def _record(self, kind: str, blk: dict, tx: dict = None) -> dict:
        """
        把一条原始记录归一成统一活动记录

        Args:
            kind: tx_from / tx_to / event
            blk:  SQD 区块对象
            tx:   交易对象（event 类可为 None）
        Returns:
            dict: 统一活动记录
        """
        h = blk.get('header', {})
        tx = tx or {}
        return {
            'timeStamp': self._safe_int(h.get('timestamp')),
            'blockNumber': self._safe_int(h.get('number')),
            'txIndex': self._safe_int(tx.get('transactionIndex')),
            'kind': kind,
            'to': (tx.get('to') or '').lower(),
            'value': self._to_dec(tx.get('value')),
            'gasPrice': self._to_dec(tx.get('gasPrice')),
            'nonce': self._safe_int(tx.get('nonce')),
            # ⚠️ SQD status(1=成功) 与 Etherscan isError(0=成功) 语义相反
            'isError': '0' if self._safe_int(tx.get('status'), 1) == 1 else '1',
            'methodId': tx.get('sighash') or '0x',
            'hash': tx.get('hash') or '',
        }

    # ============================================================
    #  扫描：三类活动
    # ============================================================
    def _segments(self) -> list:
        """
        把窗口切成 N_SEGMENTS 个等距小段 —— 🔴 2026-09-04 实测新增

        【解决什么问题】
          实测 bot 组 7 天窗口（302,400 块）的 tx_from 扫描:
              12,000 块窗口 →  94,785 条 /  45 秒
             302,400 块窗口 → 约 240 万条 / 跑 100 分钟仍未完成
          但每地址有 MAX_ACTIVITY_PER_ADDRESS(3,000) 封顶，
          150 个地址最多只留 45 万条 —— 拉回来的 99% 当场丢弃，纯浪费。

        【为什么分段而不是缩小窗口】
          缩小窗口会毁掉周期类特征: entropy_dow 需要跨度 >= 7 天
          （见 config.MIN_WINDOW_FOR_CYCLE_BLOCKS）。
          分段扫描则是「跨度不变、密度降低」:
            · 首末段仍相距 7 天    ⇒ entropy_dow / span 有效
            · 12 段散布在一周内    ⇒ 覆盖不同星期与不同时段
            · 段内区块连续         ⇒ 段内 Δt 是真实间隔
            · 段间不算 Δt          ⇒ 已有 seg 字段 + f01 的跨段过滤兜底
          这与 _merge_and_cap 的分段封顶是同一套逻辑，
          只是把「先全拉再丢弃」提前成「一开始就只拉需要的」。

        Returns:
            list: [(段号, 段起始块, 段结束块), ...]
        """
        span = max(self.hi - self.lo, 1)
        blk = min(self.SEGMENT_BLOCKS, span)
        n = max(self.N_SEGMENTS, 2)
        # ⚠️ 段起点用 (span - blk) 而非 span 做步长:
        #    否则末段起点只到 lo + 11/12·span，末段尾距窗口末端还差 1/12，
        #    实测 7 天窗口的实际跨度缩到 6.5 天 —— 刚好跌破
        #    entropy_dow 需要的一整周，白白毁掉周期特征。
        #    这样 i=0 贴窗口头、i=n-1 贴窗口尾，跨度 100% 保留。
        step = (span - blk) / (n - 1)
        # 窗口小于 N_SEGMENTS × blk 时段会互相重叠，重叠部分会被重复请求。
        # 结果不会错（_merge_and_cap 按 (ts, block, txIndex, kind) 去重），
        # 但白跑请求 —— 所以此时收窄段长，让各段刚好首尾相接。
        if step < blk:
            blk = max(int(step), 1)
        segs, seen = [], set()
        for i in range(n):
            s_lo = int(self.lo + i * step)
            s_hi = min(s_lo + blk, self.hi)
            if s_hi > s_lo and s_lo not in seen:
                seen.add(s_lo)
                segs.append((i, s_lo, s_hi))
        return segs

    def _scan_segmented(self, scan_fn, out: dict, label: str) -> int:
        """
        按段重复调用某个扫描函数，把 self.lo/hi 临时换成段边界

        Args:
            scan_fn: 无参可调用（闭包已绑定地址批），返回命中条数
            out:     累积容器（scan_fn 直接往里写）
            label:   日志用的活动类型名
        Returns:
            int: 总命中条数
        """
        base_lo, base_hi = self.lo, self.hi
        segs = self._segments()
        total = 0
        try:
            for i, (seg_no, s_lo, s_hi) in enumerate(segs):
                self.lo, self.hi = s_lo, s_hi
                total += scan_fn()
                if (i + 1) % 4 == 0 or i == len(segs) - 1:
                    log(f'    [{label}] 段 {i+1}/{len(segs)} 累计 {total:,} 条')
        finally:
            self.lo, self.hi = base_lo, base_hi
        return total

    def _scan_tx(self, addrs: list, key: str, out: dict,
                 on_progress=None) -> int:
        """
        扫 tx.from 或 tx.to

        Args:
            addrs: 地址批（≤ SQD_MAX_ADDR_PER_QUERY）
            key:   'from' / 'to'
            out:   {address: [record]} 累积容器
        Returns:
            int: 命中条数
        """
        kind = 'tx_from' if key == 'from' else 'tx_to'
        aset = {a.lower() for a in addrs}
        q = {
            'type': 'evm', 'fromBlock': self.lo, 'toBlock': self.hi,
            'fields': {
                'block': {'number': True, 'timestamp': True},
                'transaction': {'from': True, 'to': True, 'hash': True,
                                'nonce': True, 'gasPrice': True, 'value': True,
                                'status': True, 'transactionIndex': True,
                                'sighash': True},
            },
            'transactions': [{key: [a.lower() for a in addrs]}],
        }
        n = 0
        for blk in self.client.stream(q, on_progress=on_progress):
            for tx in blk.get('transactions', []):
                owner = (tx.get(key) or '').lower()
                if owner not in aset:
                    continue
                out[owner].append(self._record(kind, blk, tx))
                n += 1
        return n

    def _scan_events(self, addrs: list, out: dict, on_progress=None) -> int:
        """
        扫事件参数中出现该地址的记录（x402 付款方的主要活动形态）

        覆盖两类 USDC 事件:
          AuthorizationUsed(authorizer, nonce)  topic1 = 付款人
          Transfer(from, to, value)             topic1 = 转出方

        Args:
            addrs: 地址批
            out:   累积容器
        Returns:
            int: 命中条数
        """
        aset = {a.lower() for a in addrs}
        q = {
            'type': 'evm', 'fromBlock': self.lo, 'toBlock': self.hi,
            'fields': {
                'block': {'number': True, 'timestamp': True},
                'log': {'address': True, 'topics': True, 'data': True,
                        'transactionIndex': True},
            },
            'logs': [{'address': [self.USDC_BASE],
                      'topic0': [self.TOPIC_AUTH, self.TOPIC_TRANSFER]}],
        }
        n = 0
        for blk in self.client.stream(q, on_progress=on_progress):
            for lg in blk.get('logs', []):
                tp = lg.get('topics') or []
                if len(tp) < 2:
                    continue
                who = self._topic_to_addr(tp[1])
                if who not in aset:
                    continue
                rec = self._record('event', blk,
                                   {'transactionIndex': lg.get('transactionIndex')})
                # 事件类没有 gas/nonce（不是该地址付的 gas）
                rec['methodId'] = tp[0][:10]
                out[who].append(rec)
                n += 1
        return n

    # ============================================================
    #  合并与封顶
    # ============================================================
    # 分段数：封顶时把窗口切成几段，每段取连续记录
    N_SEGMENTS = 12
    # 分段扫描时每段扫多少块（见 config.SEGMENT_SCAN_BLOCKS 的完整说明）
    SEGMENT_BLOCKS = config.SEGMENT_SCAN_BLOCKS

    def _merge_and_cap(self, out: dict) -> dict:
        """
        合并去重 + 分段封顶（清洗逻辑 1、2）

        🔴 封顶是必须的: 单个 MEV bot 6 个月约 525 MB，700 个就是 359 GB。

        ⚠️ 但「取最后 cap 条」是错的 —— 实测 MEV bot 密度 1,093 条/4000 块，
           11,000 块就撞 3000 条上限，跨度被压缩到几小时。
           而 entropy_dow（周内节律）需要 ≥7 天 = 302,400 块的跨度
           ⇒ 节律特征全部失效。

        ★ 正确做法: **分段采样** —— 把窗口切成 N_SEGMENTS 段，
           每段取连续的 cap/N 条，并打上 seg 标记。
             · 段内连续  ⇒ Δt 有效（相邻活动间隔真实）
             · 跨段分布  ⇒ 节律特征有效（覆盖完整窗口）
             · seg 字段  ⇒ 特征计算只在段内算 Δt，跳过人为的跨段断点

        Args:
            out: {address: [record]}
        Returns:
            dict: 清洗后的 {address: [record]}（含 seg 字段）
        """
        clean, n_capped = {}, 0
        span = max(self.hi - self.lo, 1)
        seg_size = span / self.N_SEGMENTS
        per_seg = max(self.cap // self.N_SEGMENTS, 1)

        for addr, recs in out.items():
            # 2. 去重 + 排序
            seen, uniq = set(), []
            for r in recs:
                k = (r['timeStamp'], r['blockNumber'], r['txIndex'], r['kind'])
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(r)
            uniq.sort(key=lambda r: (r['timeStamp'], r['txIndex']))

            # 按区块号打段标记
            for r in uniq:
                s = int((r['blockNumber'] - self.lo) / seg_size)
                r['seg'] = min(max(s, 0), self.N_SEGMENTS - 1)

            # 1. 分段封顶：每段取前 per_seg 条连续记录
            if len(uniq) > self.cap:
                bucket, kept = defaultdict(int), []
                for r in uniq:
                    if bucket[r['seg']] < per_seg:
                        bucket[r['seg']] += 1
                        kept.append(r)
                uniq = kept
                n_capped += 1
            clean[addr] = uniq

        if n_capped:
            log(f'  {n_capped} 个地址超过 {self.cap:,} 条上限，'
                f'已按 {self.N_SEGMENTS} 段各取 {per_seg} 条（保跨度+保段内连续）', 'warn')
        self.stats['capped'] += n_capped
        return clean

    def _filter_by_chain(self, rows: list) -> list:
        """
        剔除明确属于其他链的地址 —— 🔴 2026-09-04 实测新增

        【为什么必须有】
          地址清单是跨链汇总的，source 列形如 olas_base / olas_gnosis /
          x402_payer / mev_atomic_arb。实测 fetch_olas --chains base,gnosis
          之后，agent_olas.csv 的构成变成:
              olas_base 595 / olas_gnosis 3,560 / x402_payer 617
          gnosis 占 74.6%。而活动流是在 Base 上采的 ——
          这些 Gnosis Safe 在 Base 上根本不存在，必然零活动。
          分层采样按 source 轮转，会老老实实取满 50 个 gnosis 地址，
          等于三分之一的 agent 样本注定是空的。

        【判据】 source 里出现了某个已知链名，且不是当前链 → 剔除。
                不含任何链名的 source（x402_payer / mev_atomic_arb）视为
                本链通用，保留 —— 它们本来就是在当前链上采出来的。

        Args:
            rows: 地址清单行（含 source 列）
        Returns:
            list: 只剩当前链可用的行
        """
        known = set(config.SQD_DATASETS) | set(config.CHAINS)
        others = {c for c in known if c != self.chain}
        if not others:
            return rows

        keep, dropped = [], defaultdict(int)
        for r in rows:
            src = (r.get('source') or '').lower()
            hit = next((c for c in others if c in src), None)
            if hit:
                dropped[hit] += 1
            else:
                keep.append(r)

        if dropped:
            detail = ', '.join(f'{k}={v}' for k, v in sorted(dropped.items()))
            log(f'  跨链过滤: 剔除 {sum(dropped.values()):,} 个非 {self.chain} '
                f'地址（{detail}）—— 它们在 {self.chain} 上不会有任何活动', 'warn')
        return keep

    @staticmethod
    def _stratified_sample(rows: list, n: int) -> list:
        """
        按 source 分层采样地址（清洗逻辑 5）

        ⚠️ 实测教训: 直接取前 N 个会全是 Olas —— agent_olas.csv 的构成是
           olas_base 58 / erc8004 3 / x402_payer 481，前 58 个全是 Olas Safe，
           而 Olas Safe 是休眠的合约账户（实测 4000 块内活动为 0）。
           ⇒ 必须按来源分层，否则采到的全是空的。

        Args:
            rows: 地址清单
            n:    目标采样数
        Returns:
            list: 分层采样后的地址行
        """
        if len(rows) <= n:
            return rows
        by_src = defaultdict(list)
        for r in rows:
            by_src[r.get('source', 'unknown')].append(r)

        out, srcs = [], list(by_src)
        # 轮转取，保证每个来源都有代表
        i = 0
        while len(out) < n and any(by_src.values()):
            s = srcs[i % len(srcs)]
            if by_src[s]:
                out.append(by_src[s].pop(0))
            i += 1
        dist = ', '.join(f'{k}={sum(1 for r in out if r.get("source") == k)}'
                         for k in srcs)
        log(f'  分层采样 {len(out)} 个地址: {dist}', 'info')
        return out

    def _checkpoint(self, group: str, out: dict) -> None:
        """
        每类活动扫完后增量落盘 —— 长任务的抗中断保护

        【为什么需要】 实测 human 组单类扫描 606 次请求耗时 22 分钟，
                     且遇到过 SQD 侧 IO 挂起 >12 小时。若只在末尾落盘，
                     一次挂起就会丢掉前面所有已扫完的类型。
        【注意】 这里落的是「阶段快照」，seg / win_* 字段尚未写入；
                fetch_group 末尾会用完整数据覆盖，以最终那次为准。
        """
        if not out:
            return
        try:
            snap = self._merge_and_cap({k: list(v) for k, v in out.items()})
            for addr, recs in snap.items():
                for r in recs:
                    r['win_lo'], r['win_hi'] = self.lo, self.hi
                save_cache(group, addr, self.chain, recs)
            log(f'    ↳ 检查点已落盘 {len(snap)} 个地址', 'info')
        except Exception as e:          # 检查点失败不能影响主流程
            log(f'    ↳ 检查点落盘失败（不影响采集）: {e}', 'warn')

    # ============================================================
    #  ① 单组采集
    # ============================================================
    def fetch_group(self, group: str, max_addr: int = None,
                    save: bool = True) -> dict:
        """
        ① 采集单组真值地址的统一活动流

        【数据源】 SQD stream，三类过滤器:
                 transactions[{from}] / transactions[{to}] / logs[USDC 事件]
                 - 无需 API Key，SQD 公开 Portal 免费
                 - 窗口 config.MODEL_WINDOW_START_BLOCK ~ END_BLOCK

        【指标含义】 一句话: "这个地址在链上每一次留下时间戳的时刻"
                    无论活动形式是发交易、被调用、还是授权被使用，
                    都反映「这个主体多久做一次决策」—— 这才是 Δt 该测的东西。

        【返回字段】 {address: [活动记录]}，记录字段见模块文档 二

        【阈值解读】
                    · 每地址封顶 3,000 条（超出保留最新）
                    · 活动数 < MIN_TX_FOR_FEATURES(20) 的地址会在特征阶段被剔除

        【如何使用】
                   1) 🔴 三组的活动类型构成完全不同，必须看 kind 分布:
                      MEV bot 几乎全是 tx_from；x402 付款方几乎全是 event；
                      Olas Safe 主要是 tx_to
                   2) ⚠️ tx_to / event 类活动没有 gasPrice / nonce ——
                      f_gas / f_nonce 特征对这些组会大量缺失，
                      论文必须交代这个「特征可得性差异」
                   3) 产出直接落 data/raw/{chain}/{group}/，供 compute_features 读取

        Args:
            group:    'agent' / 'bot' / 'human'
            max_addr: 最多采多少地址；缺省 config.MAX_ADDR_PER_GROUP
            save:     是否落盘
        Returns:
            dict: {address: [活动记录]}
        """
        rows = read_address_csv(config.ADDR_DIR / config.GROUPS[group][0])
        if not rows:
            log(f'[{group}] 地址清单为空，跳过。见 data/addresses/README.md', 'warn')
            return {}
        rows = self._filter_by_chain(rows)
        rows = self._stratified_sample(rows, max_addr or config.MAX_ADDR_PER_GROUP)
        addrs = [r['address'] for r in rows]

        # 🔴 按组自适应窗口 —— human 天然低频，等长窗口凑不满门槛（见 config 注释）
        base_lo, base_hi = self.lo, self.hi
        mult = config.GROUP_WINDOW_MULT.get(group, 1)
        if mult > 1:
            self.lo = max(base_hi - (base_hi - base_lo) * mult, 0)
            log(f'  [{group}] 窗口 ×{mult} 放大至 {self.hi-self.lo:,} 块'
                f'（该组天然低频，等长窗口活动量不足）', 'info')
        skip = config.GROUP_SKIP_KINDS.get(group, set())
        n_kind = 3 - len(skip)

        # 🔴 分段扫描 —— 数据量大的组只拉 12 个小段，不拉整个 7 天窗口
        seg_scan = group in config.GROUP_SEGMENT_SCAN
        if seg_scan:
            segs = self._segments()
            covered = sum(h - l for _, l, h in segs)
            log(f'  [{group}] 分段扫描: {len(segs)} 段 × {self.SEGMENT_BLOCKS:,} 块 '
                f'= {covered:,} 块（占窗口 {covered/max(self.hi-self.lo,1):.1%}）', 'info')
            log(f'    跨度仍是完整 {self.hi-self.lo:,} 块 ⇒ '
                f'周期类特征不受影响；段内连续 ⇒ 段内 Δt 有效', 'info')

        log(f'[{group}] {len(addrs)} 个地址 | 区块 {self.lo:,} → {self.hi:,}'
            f'（{self.hi-self.lo:,} 块）', 'step')
        est = (self.hi - self.lo) / config.SQD_CHUNK_BLOCKS
        log(f'  每类活动约 {est:,.0f} 次请求 × {n_kind} 类，串行约 '
            f'{est * n_kind / 5.2 / 60:.0f} 分钟', 'info')

        out = defaultdict(list)
        t0 = time.time()

        # 3. 地址分批
        batch = config.SQD_MAX_ADDR_PER_QUERY
        for i in range(0, len(addrs), batch):
            chunk = addrs[i:i + batch]
            for key, label in (('from', 'tx_from'), ('to', 'tx_to')):
                if label in skip:
                    log(f'  [{label}] 跳过（见 config.GROUP_SKIP_KINDS）')
                    continue
                if seg_scan:
                    n = self._scan_segmented(
                        lambda k=key: self._scan_tx(chunk, k, out), out, label)
                else:
                    n = self._scan_tx(chunk, key, out)
                self.stats[label] += n
                log(f'  [{label}] 命中 {n:,} 条（{time.time()-t0:.0f}s）')
                self._checkpoint(group, out)   # 每类扫完就落盘，防长任务中途丢结果
            if 'event' in skip:
                log('  [event] 跳过（见 config.GROUP_SKIP_KINDS）')
            else:
                if seg_scan:
                    n = self._scan_segmented(
                        lambda: self._scan_events(chunk, out), out, 'event')
                else:
                    n = self._scan_events(chunk, out)
                self.stats['event'] += n
                log(f'  [event] 命中 {n:,} 条（{time.time()-t0:.0f}s）')
                self._checkpoint(group, out)

        clean = self._merge_and_cap(out)
        # 把本组实际窗口写进每条记录 —— 特征端据此归一化绝对跨度特征，
        # 否则「窗口长 ⇒ human」会成为模型的作弊捷径
        for recs in clean.values():
            for r in recs:
                r['win_lo'], r['win_hi'] = self.lo, self.hi
        self.lo, self.hi = base_lo, base_hi   # 还原，不污染下一组
        n_act = sum(len(v) for v in clean.values())
        n_thin = sum(1 for v in clean.values()
                     if len(v) < config.MIN_TX_FOR_FEATURES)
        log(f'[{group}] {len(clean)} 个地址有活动，共 {n_act:,} 条；'
            f'{n_thin} 个活动数不足会被剔除', 'ok')

        if save:
            for addr, recs in clean.items():
                save_cache(group, addr, self.chain, recs)
            log(f'  已落盘 data/raw/{self.chain}/{group}/', 'ok')
        return clean

    # ============================================================
    #  ② 三组全采
    # ============================================================
    def fetch_all_groups(self, max_addr: int = None,
                         save: bool = True) -> dict:
        """
        ② 采集三组的统一活动流

        Args:
            max_addr / save
        Returns:
            dict: {group: {address: [活动记录]}}
        """
        # 🔴 单组失败不拖垮其余组 —— 实测一次本地 DNS 抖动在 bot 组开头抛出，
        #    整个进程退出，前面跑了 2 小时的 agent 组白费（幸有检查点才保住）。
        #    每组采集是独立的，一组挂掉不应影响另外两组。
        result, failed = {}, []
        for g in config.GROUPS:
            try:
                result[g] = self.fetch_group(g, max_addr=max_addr, save=save)
            except KeyboardInterrupt:
                raise                        # 用户主动中断要立刻退出
            except Exception as e:
                result[g] = {}
                failed.append((g, f'{type(e).__name__}: {str(e)[:100]}'))
                log(f'[{g}] 采集失败，跳过该组继续: {str(e)[:120]}', 'err')

        if failed:
            log('', 'info')
            log(f'🔴 {len(failed)}/{len(config.GROUPS)} 组采集失败:', 'err')
            for g, err in failed:
                log(f'   {g}: {err}')
            log('   已完成的组数据仍在 data/raw/ 中；'
                '网络恢复后用 --group <组名> 单独补跑即可', 'info')
        return result

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

    def validate_activity(self, data: dict, group: str,
                          context: str = None) -> bool:
        """
        校验单组的活动流产出

        检查项:
          1. 非空 / 必需字段齐全
          2. 时间戳落在窗口内
          3. 活动记录按时间有序
          4. ⚠️ 活动数不足门槛的地址占比
          5. ⚠️ kind 分布（三组形态应有显著差异）
          6. ⚠️ 被封顶的地址数

        Args:
            data / group / context
        Returns:
            bool
        """
        context = context or f'{group} 活动流'
        issues, warns = [], []
        need = ('timeStamp', 'blockNumber', 'txIndex', 'kind', 'to',
                'gasPrice', 'nonce', 'isError', 'methodId')

        if not data:
            issues.append('结果为空')
            return self._emit(context, issues, warns)

        sample = next((v for v in data.values() if v), None)
        if sample is None:
            issues.append('所有地址的活动列表都是空的')
            return self._emit(context, issues, warns)
        miss = [k for k in need if k not in sample[0]]
        if miss:
            issues.append(f'活动记录缺字段: {miss}')

        # 2. 时间戳窗口 —— 用记录自带的 win_lo/win_hi（各组窗口可能不等长）
        all_bn = [r['blockNumber'] for v in data.values() for r in v]
        if all_bn:
            w_lo = min((r.get('win_lo', self.lo) for v in data.values()
                        for r in v), default=self.lo)
            w_hi = max((r.get('win_hi', self.hi) for v in data.values()
                        for r in v), default=self.hi)
            if min(all_bn) < w_lo - 1 or max(all_bn) > w_hi + 1:
                issues.append(f'区块号越界: [{min(all_bn):,}, {max(all_bn):,}] '
                              f'不在 [{w_lo:,}, {w_hi:,}] 内')

        # 3. 有序性
        n_unsorted = sum(
            1 for v in data.values()
            if len(v) > 1 and any(v[i]['timeStamp'] > v[i + 1]['timeStamp']
                                  for i in range(len(v) - 1)))
        if n_unsorted:
            issues.append(f'{n_unsorted} 个地址的活动未按时间排序')

        # 4. 活动数不足
        n_thin = sum(1 for v in data.values()
                     if len(v) < config.MIN_TX_FOR_FEATURES)
        if n_thin:
            pct = n_thin / len(data)
            msg = (f'{n_thin}/{len(data)}（{pct:.0%}）个地址活动数 < '
                   f'{config.MIN_TX_FOR_FEATURES}，将在特征阶段被剔除')
            (issues if pct > 0.8 else warns).append(msg)

        # 5. kind 分布
        kinds = defaultdict(int)
        for v in data.values():
            for r in v:
                kinds[r['kind']] += 1
        tot = sum(kinds.values()) or 1
        dist = ', '.join(f'{k}={n/tot:.0%}' for k, n in sorted(kinds.items()))
        warns.append(f'活动类型构成: {dist}（{tot:,} 条）')
        if kinds.get('tx_from', 0) / tot < 0.05:
            warns.append('⚠️ tx_from 占比 <5% —— 该组 gasPrice/nonce 特征将大量缺失，'
                         '论文需交代此「特征可得性差异」')

        # 6. 封顶
        n_cap = sum(1 for v in data.values() if len(v) >= self.cap)
        if n_cap:
            warns.append(f'{n_cap} 个地址达到 {self.cap:,} 条封顶，'
                         f'已保留最新部分（不影响行为统计量）')
        return self._emit(context, issues, warns)


# ================================================================
#  ActivityFetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class ActivityFetcher(BaseFetcher):
    """
    薄适配层：把 ActivityDataSource 接进项目的 Fetcher 体系。

    ⚠️ 产出落到 data/raw/ 而非 data/sources/。
    """

    name = 'activity'
    desc = '统一活动流（特征计算的直接输入）'
    output = ''
    needs_key = False

    def __init__(self, group='all', max_addr=None, blocks=0,
                 lo=None, hi=None, cap=None, **kw):
        super().__init__(**kw)
        self.group = group
        self.max_addr = int(max_addr) if max_addr else None
        # --blocks N 表示只扫窗口末尾 N 块（快速验证用）
        _hi = hi or config.MODEL_WINDOW_END_BLOCK
        _lo = lo or (max(config.MODEL_WINDOW_START_BLOCK, _hi - int(blocks))
                     if blocks else config.MODEL_WINDOW_START_BLOCK)
        self.ds = ActivityDataSource(lo=_lo, hi=_hi, cap=cap)
        self.data = {}

    def fetch(self) -> list:
        groups = list(config.GROUPS) if self.group == 'all' else [self.group]
        total = 0
        for g in groups:
            d = self.ds.fetch_group(g, max_addr=self.max_addr, save=True)
            if d:
                self.ds.validate_activity(d, g)
                self.data[g] = d
                total += len(d)
        for k, v in self.ds.stats.items():
            self.stats[f'活动_{k}'] = f'{v:,}'
        self.stats['采集地址总数'] = total
        if total:
            log('下一步：python3 src/feature/compute_features.py', 'ok')
        return []

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        log(f'窗口 {self.ds.lo:,} → {self.ds.hi:,} | 封顶 {self.ds.cap:,} 条/地址',
            'info')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--group', default='all',
                        choices=['all'] + list(config.GROUPS))
        ap.add_argument('--max-addr', type=int, default=None,
                        help='每组最多采多少地址（缺省 config.MAX_ADDR_PER_GROUP）')
        ap.add_argument('--blocks', type=int, default=0,
                        help='只扫窗口末尾 N 块（0=全窗口；快速验证建议 20000）')
        ap.add_argument('--cap', type=int, default=None,
                        help='每地址活动上限（缺省 config.MAX_ACTIVITY_PER_ADDRESS）')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(description='统一活动流采集（三组真值地址）')
    ActivityFetcher.add_args(ap)
    args = ap.parse_args()
    ActivityFetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
