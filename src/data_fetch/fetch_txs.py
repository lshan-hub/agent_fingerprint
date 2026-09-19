#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_txs.py
@Description:
    地址交易史拉取（双后端）—— 特征计算的输入，附 validate_* 校验。

    ============================================================
    一、 认证与限频
    ============================================================
    两个后端，各有最优场景（2026-09-03 实测结论）:

      · Etherscan V2 —— 按地址建了索引
          Base URL: https://api.etherscan.io/v2/api
          认证:     apikey 参数（环境变量 ETHERSCAN_API_KEY > config.ETHERSCAN_API_KEY）
          限频:     免费层 5 req/s（全链共享）、100,000 req/天
          ⚠️ 2026-07-01 起免费层单页上限 10000 → 1000 条，本模块已分页
          ⚠️ 免费层链覆盖降至约 90%
          → go/no-go 阶段（600 个已知地址）推荐用它，几分钟跑完

      · SQD Network 公开 Portal —— 无地址索引但完全免费
          Base URL: https://portal.sqd.dev
          认证:     无 —— 无需 API Key、无需注册
          限频:     无硬限流；单次响应只推进约 396 区块
          实测:     8 并发最优 5.22 req/s；16 并发降到 3.09 req/s 且大量 529
          → 主实验（1200+ 地址）和 RQ3 全网扫描必须用它
             1000 个地址可一次过滤 ⇒ 地址越多越划算

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — Etherscan V2 账户交易列表 ────────────────────────────────────┐
    │  接口:  GET /v2/api?chainid={id}&module=account&action=txlist&address=... │
    │  函数:  fetch_group(backend='etherscan')                                  │
    │  更新频率: 实时 │ API KEY: 是（免费）                                     │  ★必填
    │  历史范围: 全历史；⚠️ 单页 1000 条，本模块最多翻 20 页（2 万笔/地址）     │  ★必填
    │  产出: data/raw/{chain}/{group}/{address}.json                            │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 🔴 坑: 填 offset=10000 **不会报错，而是静默只返回 1000 条** ——
        这种失败最危险，会让你以为拿全了。本模块已改为分页。

    ┌─ 数据源② — SQD Portal 批量地址过滤 ──────────────────────────────────────┐
    │  接口:  POST /datasets/{ds}/stream，transactions 过滤器指定 from 地址列表 │
    │  函数:  fetch_group(backend='sqd')                                        │
    │  更新频率: 实时 │ API KEY: 否                                             │  ★必填
    │  历史范围: ⭐ 全历史（start_block=0）；窗口由 config.WINDOW_*_BLOCK 控制  │  ★必填
    │  产出: data/raw/{chain}/{group}/{address}.json                            │
    └───────────────────────────────────────────────────────────────────────────┘

      两后端字段已统一（见 SQDClient._normalize）:
        字段          Etherscan 原始      SQD 原始         统一后
        blockNumber   十进制字符串         0x 十六进制       int
        timeStamp     十进制字符串         十进制           int
        gasPrice      十进制字符串         0x 十六进制       十进制字符串
        成功标志       isError(0=成功)     status(1=成功)   isError（语义相反，已转换）
        methodId      input[:10]          sighash          methodId

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. 结果落盘缓存到 data/raw/{chain}/{group}/{address}.json，重跑不重复消耗额度
      2. 交易数 < config.MIN_TX_FOR_FEATURES(20) 的地址会在特征计算阶段被剔除，
         本模块只统计不删除（保留原始数据便于复查）
      3. 单地址失败不中断整批，记录到 n_fail

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🔗 **管线枢纽** —— 上承 data/addresses/ 的三组真值清单，
        下接 src/feature/compute_features.py 的特征计算。
        本身不产出 data/sources/ 的 CSV，而是把原始交易落到 data/raw/。
'''

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils._base import BaseFetcher, log  # noqa: E402
from src.utils.common import (etherscan_get, load_cache,  # noqa: E402
                              read_address_csv, save_cache)


# ================================================================
#  TxHistoryDataSource — 地址交易史（双后端）
# ================================================================
class TxHistoryDataSource(object):
    """
    地址交易史采集与管理（Etherscan V2 / SQD 双后端）

    核心函数 (共 1 个，双后端):
      ① fetch_group(group, backend)
          backend='etherscan' → 按地址索引，少量已知地址更快
          backend='sqd'       → 免费全历史，大批量与全网扫描

    辅助函数:
      - _slim()                  裁剪 Etherscan 返回，只留下游要用的字段
      - _fetch_one_etherscan()   单地址翻页拉取
      - _fetch_group_etherscan() / _fetch_group_sqd()

    校验函数: validate_group

    Args:
        chain:   目标链名
        refresh: True 时忽略缓存重新拉取
    """

    def __init__(self, chain: str = None, refresh: bool = False):
        """
        初始化

        Args:
            chain:   目标链名；缺省 config.PRIMARY_CHAIN
            refresh: 是否忽略缓存
        Returns:
            None
        """
        self.chain = chain or config.PRIMARY_CHAIN
        self.refresh = bool(refresh)
        self.n_fail = 0
        self.n_cached = 0
        self.n_thin = 0

    # ============================================================
    #  通用工具
    # ============================================================
    @staticmethod
    def _safe_int(v, default: int = 0):
        """安全转 int（支持 0x 前缀十六进制）。"""
        try:
            if v is None or v == '':
                return default
            return int(v, 16) if isinstance(v, str) and v.startswith('0x') else int(v)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _slim(cls, t: dict) -> dict:
        """裁剪 Etherscan 返回，只留下游特征计算要用的字段。"""
        return {
            'hash': t.get('hash'),
            'blockNumber': cls._safe_int(t.get('blockNumber')),
            'timeStamp': cls._safe_int(t.get('timeStamp')),
            'to': (t.get('to') or '').lower(),
            'value': t.get('value', '0'),
            'gasPrice': t.get('gasPrice', '0'),
            'gasUsed': t.get('gasUsed', '0'),
            'nonce': cls._safe_int(t.get('nonce')),
            'isError': t.get('isError', '0'),
            'methodId': (t.get('input') or '0x')[:10],
            'txIndex': cls._safe_int(t.get('transactionIndex')),
        }

    def _read_group_addresses(self, group: str) -> list:
        """读某一组的真值地址清单。"""
        rows = read_address_csv(config.ADDR_DIR / config.GROUPS[group][0])
        if not rows:
            log(f'[{group}] 地址清单为空，跳过。见 data/addresses/README.md', 'warn')
            return []
        return rows[: config.MAX_ADDR_PER_GROUP]

    # ============================================================
    #  ① 后端 A：Etherscan V2
    # ============================================================
    def _fetch_one_etherscan(self, address: str) -> list:
        """
        翻页拉取单地址全部交易

        ⚠️ 2026-07-01 起免费层单页上限 1000 条。填 offset=10000 不会报错，
           而是静默只返回 1000 条 —— 这种失败最危险。

        Args:
            address: 目标地址
        Returns:
            list[dict]: 已裁剪的交易列表
        """
        out = []
        for page in range(1, config.ETHERSCAN_MAX_PAGES + 1):
            data = etherscan_get(self.chain, {
                'module': 'account', 'action': 'txlist', 'address': address,
                'startblock': config.START_BLOCK, 'endblock': config.END_BLOCK,
                'page': page, 'offset': config.ETHERSCAN_PAGE_SIZE, 'sort': 'asc',
            })
            status, result = str(data.get('status', '')), data.get('result', [])

            if status != '1' or not isinstance(result, list):
                if 'No transactions found' in str(data.get('message', '')):
                    break
                if page == 1:
                    raise RuntimeError(f"{address}: {data.get('message')} / {result}")
                break

            out.extend(self._slim(t) for t in result)
            if len(result) < config.ETHERSCAN_PAGE_SIZE:
                break   # 最后一页
        else:
            log(f'  {address} 达到 {config.ETHERSCAN_MAX_PAGES} 页上限，可能被截断',
                'warn')
        return out

    def _fetch_group_etherscan(self, group: str) -> dict:
        """按地址逐个拉取（Etherscan 有索引，少量地址更快）。"""
        rows = self._read_group_addresses(group)
        if not rows:
            return {}
        log(f'[{group}] {len(rows)} 个地址，链={self.chain}', 'step')

        out = {}
        for i, row in enumerate(rows, 1):
            addr = row['address']
            if not self.refresh:
                c = load_cache(group, addr, self.chain)
                if c is not None:
                    out[addr] = c
                    self.n_cached += 1
                    continue
            try:
                txs = self._fetch_one_etherscan(addr)
            except Exception as e:
                self.n_fail += 1
                log(f'  {addr} 失败: {e}', 'err')
                continue
            save_cache(group, addr, self.chain, txs)      # 1. 落盘缓存
            out[addr] = txs
            if len(txs) < config.MIN_TX_FOR_FEATURES:     # 2. 只统计不删除
                self.n_thin += 1
            if i % 25 == 0 or i == len(rows):
                log(f'  进度 {i}/{len(rows)}（缓存命中 {self.n_cached}）')

        log(f'[{group}] {len(out)} 个地址，{self.n_thin} 个交易数不足，'
            f'{self.n_fail} 个失败', 'ok')
        return out

    # ============================================================
    #  ② 后端 B：SQD Portal
    # ============================================================
    def _fetch_group_sqd(self, group: str) -> dict:
        """批量地址过滤（SQD 一次可过滤 1000 个地址，地址越多越划算）。"""
        from src.utils.sqd_client import SQDClient

        rows = self._read_group_addresses(group)
        if not rows:
            return {}

        pending = [r['address'] for r in rows
                   if self.refresh or load_cache(group, r['address'],
                                                 self.chain) is None]
        cached = {r['address']: load_cache(group, r['address'], self.chain)
                  for r in rows if r['address'] not in pending}
        cached = {k: v for k, v in cached.items() if v is not None}
        self.n_cached = len(cached)

        if not pending:
            log(f'[{group}] {len(cached)} 个地址全部命中缓存', 'ok')
            return cached

        log(f'[{group}] {len(pending)} 个待拉 / {len(cached)} 个命中缓存', 'step')
        est = ((config.WINDOW_END_BLOCK - config.WINDOW_START_BLOCK)
               / config.SQD_CHUNK_BLOCKS)
        log(f'  预计约 {est:,.0f} 次请求，8 并发下约 {est / 5.2 / 60:.0f} 分钟', 'info')

        client = SQDClient()

        def progress(done, total):
            if done % 10 == 0 or done == total:
                log(f'  分段 {done}/{total}')

        fetched = client.fetch_addresses(
            pending, config.WINDOW_START_BLOCK, config.WINDOW_END_BLOCK,
            on_progress=progress)

        for addr, txs in fetched.items():
            save_cache(group, addr, self.chain, txs)      # 1. 落盘缓存

        out = {**cached, **fetched}
        self.n_thin = sum(1 for v in out.values()
                          if len(v) < config.MIN_TX_FOR_FEATURES)
        log(f'[{group}] {len(out)} 个地址，其中 {self.n_thin} 个交易数不足会被剔除',
            'ok')
        return out

    def fetch_group(self, group: str, backend: str = 'etherscan') -> dict:
        """
        ① 拉取某一组真值地址的交易历史

        【数据源】 backend='etherscan' → GET /v2/api?...&action=txlist（按地址索引）
                 backend='sqd'       → POST /datasets/{ds}/stream（批量过滤）

        【指标含义】 一句话: "特征计算的原始输入"
                    三组真值地址（agent / bot / human）各自的完整交易序列，
                    是后续所有行为特征（延迟/节律/gas/nonce）的唯一数据来源。

        【返回字段】 dict: {address: [tx, ...]}
                    每个 tx 含 hash / blockNumber / timeStamp / to / value /
                    gasPrice / gasUsed / nonce / isError / methodId / txIndex

        【阈值解读】 交易数 < config.MIN_TX_FOR_FEATURES(20) 的地址，
                    统计量不可靠，会在特征计算阶段被剔除。

        【如何使用】
                   1) go/no-go 阶段（几百个已知地址）→ backend='etherscan'，几分钟
                   2) 主实验（1200+ 地址）与全网扫描 → backend='sqd'，免费无限制
                   3) 🔴 Etherscan 单页上限 1000 条，本模块已分页；
                      若自己写查询务必注意「填 10000 会静默截断」这个坑

        Args:
            group:   'agent' / 'bot' / 'human'
            backend: 'etherscan' / 'sqd'
        Returns:
            dict: {address: [tx, ...]}
        """
        self.n_fail = self.n_cached = self.n_thin = 0
        return (self._fetch_group_sqd(group) if backend == 'sqd'
                else self._fetch_group_etherscan(group))

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

    def validate_group(self, data: dict, group: str,
                       context: str = None) -> bool:
        """
        校验某一组的交易史产出

        检查项:
          1. 非空
          2. 必需字段齐全（抽样首个地址的首笔交易）
          3. 时间戳合理（2020 年之后、不在未来）
          4. ⚠️ 交易数不足门槛的地址占比
          5. ⚠️ 是否有地址疑似被 Etherscan 页数上限截断

        Args:
            data:    fetch_group 的结果
            group:   组名
            context: 日志上下文名；缺省用组名
        Returns:
            bool
        """
        import time as _t
        context = context or f'{group} 交易史'
        issues, warns = [], []
        need = ('blockNumber', 'timeStamp', 'to', 'gasPrice', 'nonce',
                'isError', 'methodId')

        if not data:
            issues.append('结果为空')
            return self._emit(context, issues, warns)

        sample = next((v for v in data.values() if v), None)
        if sample is None:
            issues.append('所有地址的交易列表都是空的')
            return self._emit(context, issues, warns)
        miss = [k for k in need if k not in sample[0]]
        if miss:
            issues.append(f'交易缺字段: {miss}')

        now = int(_t.time())
        all_ts = [t['timeStamp'] for v in data.values() for t in v
                  if isinstance(t.get('timeStamp'), int)]
        if all_ts:
            if max(all_ts) > now + 86400:
                issues.append('存在未来时间戳')
            if min(all_ts) < 1577836800:      # 2020-01-01
                warns.append('存在 2020 年之前的时间戳，请核实口径')

        n_thin = sum(1 for v in data.values() if len(v) < config.MIN_TX_FOR_FEATURES)
        if n_thin:
            pct = n_thin / len(data)
            msg = (f'{n_thin}/{len(data)}（{pct:.0%}）个地址交易数 < '
                   f'{config.MIN_TX_FOR_FEATURES}，将在特征计算阶段被剔除')
            (issues if pct > 0.5 else warns).append(msg)

        cap = config.ETHERSCAN_PAGE_SIZE * config.ETHERSCAN_MAX_PAGES
        n_cap = sum(1 for v in data.values() if len(v) >= cap)
        if n_cap:
            warns.append(f'{n_cap} 个地址达到 {cap:,} 笔上限，可能被截断 —— '
                         f'高频地址建议改用 SQD 后端')
        return self._emit(context, issues, warns)


# ================================================================
#  TxHistoryFetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class TxHistoryFetcher(BaseFetcher):
    """
    薄适配层：把 TxHistoryDataSource 接进项目的 Fetcher 体系。

    ⚠️ 与其他 Fetcher 不同，本类产出落到 data/raw/ 而非 data/sources/。
    """

    name = 'txs'
    desc = '地址交易史（Etherscan / SQD 双后端）'
    output = ''          # 不写 sources CSV
    needs_key = False    # sqd 后端不需要；etherscan 需要

    def __init__(self, backend='etherscan', chain=None, group='all',
                 refresh=False, selftest_only=False, **kw):
        super().__init__(**kw)
        self.backend = backend
        self.chain = chain or config.PRIMARY_CHAIN
        self.group = group
        self.selftest_only = bool(selftest_only)
        self.ds = TxHistoryDataSource(chain=self.chain, refresh=refresh)

    def fetch(self) -> list:
        if self.selftest_only:
            from src.utils.sqd_client import selftest
            selftest()
            return []

        if self.backend == 'sqd' and self.chain != 'base':
            log(f'SQD 数据集当前配置为 {config.SQD_DATASET}，'
                f'换链请同步改 config.SQD_DATASET', 'warn')

        log(f'后端: {self.backend} | 链: {self.chain}'
            f'（出块 {config.BLOCK_TIME[self.chain]}s）', 'info')
        if config.BLOCK_TIME[self.chain] > 3:
            log(f'出块 {config.BLOCK_TIME[self.chain]}s > 3s，分辨率不足以区分 '
                f'bot(<2s) 与 agent(2-10s)。主实验请用 base。', 'warn')

        groups = list(config.GROUPS) if self.group == 'all' else [self.group]
        total = 0
        for g in groups:
            data = self.ds.fetch_group(g, self.backend)
            if data:
                self.ds.validate_group(data, g)
            total += len(data)

        self.stats['拉取地址总数'] = total
        if total == 0:
            log('三组都没有数据。先填 data/addresses/*.csv', 'err')
        else:
            log(f'完成，共 {total} 个地址。'
                f'下一步：python3 src/feature/compute_features.py', 'ok')
        return []      # 数据已落 data/raw/，不返回行

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--backend', default='etherscan',
                        choices=['etherscan', 'sqd'],
                        help='etherscan=已知地址更快；sqd=免费全历史，大批量更划算')
        ap.add_argument('--chain', default=config.PRIMARY_CHAIN,
                        choices=list(config.CHAINS))
        ap.add_argument('--group', default='all',
                        choices=['all'] + list(config.GROUPS))
        ap.add_argument('--refresh', action='store_true', help='忽略缓存重新拉取')
        ap.add_argument('--selftest-only', action='store_true',
                        help='只跑 SQD 连通性自检')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(description='地址交易史拉取（Etherscan / SQD 双后端）')
    TxHistoryFetcher.add_args(ap)
    args = ap.parse_args()
    TxHistoryFetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
