#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_erc8004.py
@Description:
    ERC-8004 AgentIdentity 注册表链上直读 —— agent 正样本（须分层），附 validate_* 校验。

    ============================================================
    一、 认证与限频
    ============================================================
      方式:   链上直读，无需 API Key、无需注册、零成本
      RPC:    src/utils/_base.RPC_POOL（多端点轮询 + 失败拉黑 + 指数退避）
      限频:   ⚠️ 公共 RPC 限流严格，RpcPool 默认 0.12s 间隔 + 多端点轮询

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — ERC-8004 IdentityRegistry (agent 身份注册表) ⚠️须分层 ────────┐
    │  接口:  eth_call ownerOf(agentId)   → 控制钱包地址                        │
    │         eth_call tokenURI(agentId)  → 元数据 URI（服务声明）              │
    │         eth_call totalSupply()      → ⚠️ Base 上会 revert，已自动降级     │
    │  函数:  fetch_identities() (内部 _probe / _scan / _stratify)              │
    │  覆盖:  30+ 条主网，合约地址全部相同                                       │
    │  更新频率: 实时（随链上注册行为）                                          │  ★必填
    │  更新时间: 无固定刷新点                                                    │  ★必填
    │  API KEY: 否 —— 链上直读，完全免费                                        │  ★必填
    │  历史范围: 当前全量快照（递增枚举 agentId）                                │  ★必填
    │  产出: data/sources/erc8004_{chain}.csv                                   │
    └───────────────────────────────────────────────────────────────────────────┘
      合约（已链上验证 name()="AgentIdentity" symbol()="AGENT"）:
        IdentityRegistry    0x8004A169FB4a3325136EB29fA0ceB6D2e539a432
        ReputationRegistry  0x8004BAa17C55a88189AE136b182e5fdA19dE9b63

      ※ 🔴 【绝对不能直接当正样本】——全网注册量 38.6 万，但实证有效注册率仅
        Ethereum 3% / BSC 4% / Base 15%；某研究抽 1 万个逐项核查，
        **四项俱全的只有 19 个（0.19%）**。注册表 99.8% 是空壳。
      ※ ⚠️ 该合约在 Base 上是 130 字节代理，totalSupply() 会 revert，
        脚本自动降级为「递增探测 + 连续 5 次失败即停」。

      链上原始返回     → 统一字段    → 说明
      ownerOf(id)     → address     控制钱包（★候选正样本）
      tokenURI(id)    → tokenURI    元数据 URI，空 = 无服务声明
      [遍历下标]       → note        agentId=N [+ uri=yes/EMPTY]
      eth_getTxCount  → tx_count    该 owner 的历史交易数（活跃度代理）
      [客户端合成]     → tier        active / dormant / unknown

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. owner 为零地址 ⇒ 跳过
      2. ★ 活跃度分层（必做，不是可选）：
         ⚠️ 实测教训 —— 不能用「nonce > 0」当活跃判据。
            注册 ERC-8004 身份本身就是一笔交易，所以每个 owner 的 nonce 必然 > 0
            —— 实测前 40 个 owner 100% 命中，这个判据等于没分层。
         改用两条更严格的判据（任一不满足即 dormant）：
            ① tx_count >= ACTIVE_TX_THRESHOLD (20) —— 有超出注册行为的实际运营
            ② tokenURI 非空（若本次抓了 URI）      —— 有服务声明
      3. ⚠️ 上述仍只是代理指标。论文级分层还需验证 tokenURI 端点是否存活
         —— 已有研究正是这么做的，结论是有效注册率仅 3–15%。

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🥉 agent 正样本的**补充来源**，真值质量 ★★（须分层后才可用）。
        量大但噪声极高，只能取 tier=active 子集；dormant 部分可单列做
        「注册重、运营浅」的现象分析 —— 这本身就是论文里一节有价值的实证。
'''

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils._base import (BaseFetcher, RpcPool, dec_string,  # noqa: E402
                             enc_uint, log, selector, word_to_addr, words,
                             write_csv)


# ================================================================
#  Erc8004DataSource — ERC-8004 注册表链上直读
# ================================================================
class Erc8004DataSource(object):
    """
    ERC-8004 AgentIdentity 注册表数据采集与管理

    核心函数 (共 1 个):
      ① fetch_identities(chain, limit, with_uri, stratify)
          eth_call ownerOf / tokenURI → 控制钱包 + 活跃度分层

    辅助函数:
      - _probe()      确认合约存在并读元数据（含 totalSupply 降级）
      - _scan()       递增枚举 agentId
      - _stratify()   活跃度分层（active / dormant）

    校验函数: validate_identities

    Args:
        chain: 目标链名（须在 _base.RPC_POOL 中有配置）
    """

    IDENTITY = '0x8004A169FB4a3325136EB29fA0ceB6D2e539a432'
    REPUTATION = '0x8004BAa17C55a88189AE136b182e5fdA19dE9b63'

    SEL_TOTAL = selector('totalSupply()')
    SEL_OWNER = selector('ownerOf(uint256)')
    SEL_URI = selector('tokenURI(uint256)')
    SEL_NAME = selector('name()')

    # 超过「注册」本身所需的最低交易数
    ACTIVE_TX_THRESHOLD = 20
    # totalSupply 不可用时，连续多少个 agentId 取不到就停止探测
    MISS_STOP = 5
    # 文献报告的有效注册率上界，用于反向校验本地判据是否偏松
    LITERATURE_ACTIVE_MAX = 0.15

    ADDR_RE = re.compile(r'^0x[0-9a-f]{40}$')

    def __init__(self, chain: str = 'base'):
        """
        初始化

        Args:
            chain: 目标链名
        Returns:
            None
        """
        self.chain = chain
        self.pool = None
        self.n_calls = 0

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

    def _ensure_pool(self):
        if self.pool is None:
            self.pool = RpcPool(self.chain)
        return self.pool

    def _probe(self):
        """
        确认合约存在并读元数据

        ⚠️ 有些部署（如 Base 的 130 字节代理）没有 totalSupply，需容错降级。

        Returns:
            int | None: 注册总量；None 表示需改用递增探测
        """
        pool = self._ensure_pool()
        code = pool.call('eth_getCode', [self.IDENTITY, 'latest'])
        if len(code) <= 2:
            raise RuntimeError(f'{self.IDENTITY} 在 {self.chain} 上没有代码')
        log(f'IdentityRegistry 存在，bytecode {len(code)//2-1} 字节', 'ok')

        try:
            nm = dec_string(pool.call(
                'eth_call', [{'to': self.IDENTITY, 'data': self.SEL_NAME}, 'latest']))
            log(f'  name() = {nm}')
        except Exception:
            pass

        try:
            n = self._safe_int(pool.call(
                'eth_call', [{'to': self.IDENTITY, 'data': self.SEL_TOTAL}, 'latest']))
            log(f'  totalSupply() = {n:,} 个已注册身份', 'ok')
            return n
        except Exception as e:
            log(f'  totalSupply() 不可用（{str(e)[:40]}），改用递增探测', 'warn')
            return None

    def _scan(self, total, limit: int, with_uri: bool) -> list:
        """
        递增枚举 agentId，取 owner（+可选 tokenURI）

        Args:
            total:    注册总量；None 时探测到连续失败为止
            limit:    最多扫多少个 agentId
            with_uri: 是否同时取 tokenURI（慢一倍）
        Returns:
            list[dict]
        """
        pool = self._ensure_pool()
        rows, miss = [], 0
        n = min(total, limit) if (total and limit) else (total or limit)

        for aid in range(1, n + 1):
            try:
                owner = word_to_addr(words(pool.call(
                    'eth_call',
                    [{'to': self.IDENTITY,
                      'data': self.SEL_OWNER + enc_uint(aid)}, 'latest']))[0])
                miss = 0
            except Exception:
                miss += 1
                if total is None and miss >= self.MISS_STOP:
                    log(f'  连续 {miss} 个 agentId 不存在，停止于 #{aid}', 'info')
                    break
                continue

            if int(owner, 16) == 0:      # 1. 零地址跳过
                continue

            row = {'address': owner.lower(), 'source': 'erc8004',
                   'note': f'agentId={aid}'}
            if with_uri:
                try:
                    uri = dec_string(pool.call(
                        'eth_call',
                        [{'to': self.IDENTITY,
                          'data': self.SEL_URI + enc_uint(aid)}, 'latest']))
                    row['tokenURI'] = uri[:180]
                    row['note'] += f" uri={'yes' if uri else 'EMPTY'}"
                except Exception:
                    row['tokenURI'] = ''
            rows.append(row)

            if aid % 50 == 0:
                log(f'  进度 {aid}/{n} · 已得 {len(rows)} 个 owner · RPC {pool.n_calls}')
        return rows

    def _stratify(self, rows: list) -> list:
        """
        活跃度分层（清洗逻辑 2）—— 这一步是必做的，不是可选

        ⚠️ 实测教训: 不能用「nonce > 0」当活跃判据。注册身份本身就是一笔交易，
           所以每个 owner 的 nonce 必然 > 0 —— 实测前 40 个 100% 命中，等于没分层。

        Args:
            rows: _scan 的结果
        Returns:
            list[dict]: 每行补上 tx_count 与 tier
        """
        pool = self._ensure_pool()
        log(f'活跃度分层（判据：tx_count ≥ {self.ACTIVE_TX_THRESHOLD}'
            f'，且 tokenURI 非空若已抓取）...', 'step')

        seen, n_active = {}, 0
        for r in rows:
            a = r['address']
            if a not in seen:
                try:
                    seen[a] = self._safe_int(
                        pool.call('eth_getTransactionCount', [a, 'latest']), -1)
                except Exception:
                    seen[a] = -1
            nc = seen[a]
            r['tx_count'] = nc

            if nc < 0:
                r['tier'] = 'unknown'
            elif nc < self.ACTIVE_TX_THRESHOLD:
                r['tier'] = 'dormant'
            elif 'tokenURI' in r and not r['tokenURI']:
                r['tier'] = 'dormant'      # 有交易但无服务声明
            else:
                r['tier'] = 'active'
                n_active += 1

        pct = n_active / max(len(rows), 1)
        log(f'  active {n_active}/{len(rows)}（{pct:.1%}）', 'ok')
        if pct > self.LITERATURE_ACTIVE_MAX * 3:
            log(f'  ⚠️ 活跃比例明显高于文献报告的 3–15% —— '
                f'说明本代理判据仍偏宽松，论文中需用 tokenURI 端点存活性复核', 'warn')
        return rows

    # ============================================================
    #  ① ERC-8004 注册表 → agent 候选正样本
    # ============================================================
    def fetch_identities(self, limit: int = 200, with_uri: bool = False,
                         stratify: bool = True,
                         save_to_csv: bool = True) -> pd.DataFrame:
        """
        ① 枚举 ERC-8004 注册表并做活跃度分层

        【数据源】 eth_call ownerOf(agentId)  → 控制钱包
                 eth_call tokenURI(agentId) → 元数据 URI
                 - 无需 API Key，链上直读，零成本
                 - 30+ 条主网合约地址相同: 0x8004A169FB4a3325136EB29fA0ceB6D2e539a432
                 - ⚠️ Base 上 totalSupply() 会 revert（130 字节代理），已自动降级递增探测

        【指标含义】 一句话: "声称自己是 AI agent 的链上身份"
                    ERC-8004 是 agent 的身份注册标准（ERC-721，agentId = tokenId）。
                    🔴 但它是**自报身份**，不是行为证据 ——
                       实证有效注册率仅 3–15%，1 万个里只有 19 个四项俱全（0.19%）。

        【返回字段】
                  · address   — 控制钱包地址（小写）
                  · source    — erc8004
                  · note      — agentId=N [+ uri=yes/EMPTY]
                  · tx_count  — 该 owner 的历史交易数（活跃度代理）
                  · tier      — active / dormant / unknown
                  · tokenURI  — 元数据 URI（--with-uri 时）

        【阈值解读】 (tier 判定)
                    · tx_count >= 20 且 tokenURI 非空 → active（可作正样本）
                    · 其余                            → dormant（空壳，应排除）
                    · 文献口径的有效注册率: ETH 3% / BSC 4% / Base 15%
                      若本地 active 率远高于此，说明判据偏松

        【如何使用】
                   1) 🔴 只用 tier=active 子集作正样本，dormant 必须排除
                   2) dormant 部分可单列做「注册重、运营浅」的现象分析 ——
                      这本身就是论文里一节有价值的实证
                   3) 与 Olas（高置信）分层使用，训练时用 PU learning 处理噪声
                   4) 论文级分层还需验证 tokenURI 端点存活性，本脚本未做

        Args:
            limit:       最多扫多少个 agentId
            with_uri:    是否同时取 tokenURI（慢一倍，但分层更准）
            stratify:    是否做活跃度分层
            save_to_csv: 是否写入 data/sources/erc8004_{chain}.csv
        Returns:
            pd.DataFrame
        """
        log(f'链: {self.chain} | IdentityRegistry {self.IDENTITY}', 'step')
        total = self._probe()

        rows = self._scan(total, limit, with_uri)
        if not rows:
            log('没有扫到任何注册身份。该链可能尚未部署或注册量为 0。', 'warn')
            return pd.DataFrame()

        if stratify:
            rows = self._stratify(rows)

        df = pd.DataFrame(rows)
        cols = ['address', 'source', 'note']
        if stratify:
            cols += ['tier', 'tx_count']
        if with_uri:
            cols += ['tokenURI']
        df = df[[c for c in cols if c in df.columns]]

        self.n_calls += self.pool.n_calls if self.pool else 0

        if stratify:
            n_act = int((df['tier'] == 'active').sum())
            log(f'⚠️ 建议只用 tier=active 的 {n_act} 个作正样本，'
                f'dormant 的 {len(df)-n_act} 个应排除或单列分析', 'warn')

        if save_to_csv:
            write_csv(f'erc8004_{self.chain}.csv', df.to_dict('records'),
                      list(df.columns),
                      note=('ERC-8004 AgentIdentity 注册表的 owner 地址\n'
                            '🔴 有效注册率仅 3-15%，务必只用 tier=active 子集作正样本\n'
                            f'链: {self.chain} | 扫描上限 {limit}'))
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

    def validate_identities(self, df: pd.DataFrame,
                            context: str = 'ERC-8004 注册表') -> bool:
        """
        校验 ① 的产出

        检查项:
          1. 非空 / 必需列齐全
          2. 地址格式合法
          3. tier 取值合法
          4. ⚠️ active 率是否远高于文献口径（判据偏松的信号）
          5. ⚠️ 是否有 unknown（RPC 失败）
          6. 🔴 强制提示: 只能用 active 子集

        Args:
            df / context
        Returns:
            bool
        """
        issues, warns = [], []

        if df is None or df.empty:
            issues.append('结果为空')
            return self._emit(context, issues, warns)

        miss = [c for c in ('address', 'source', 'note') if c not in df.columns]
        if miss:
            issues.append(f'缺列: {miss}')
            return self._emit(context, issues, warns)

        bad = df[~df['address'].astype(str).str.match(self.ADDR_RE)]
        if len(bad):
            issues.append(f'{len(bad)} 个地址格式非法')

        if 'tier' in df.columns:
            illegal = set(df['tier'].unique()) - {'active', 'dormant', 'unknown'}
            if illegal:
                issues.append(f'tier 出现非法取值: {illegal}')

            n_unknown = int((df['tier'] == 'unknown').sum())
            if n_unknown:
                warns.append(f'{n_unknown} 个 tier=unknown（RPC 取 nonce 失败），'
                             f'这些地址无法判定，建议重跑或排除')

            act_rate = (df['tier'] == 'active').mean()
            if act_rate > self.LITERATURE_ACTIVE_MAX * 3:
                warns.append(f'active 率 {act_rate:.1%} 远高于文献口径 3–15% '
                             f'—— 代理判据偏松，论文中需用 tokenURI 端点存活性复核')
            n_act = int((df['tier'] == 'active').sum())
            if n_act < 30:
                warns.append(f'active 仅 {n_act} 个，作正样本偏少，'
                             f'建议加大 --limit 或改用 Olas 作主力')
        else:
            warns.append('未做活跃度分层 —— 🔴 这样的数据不能直接作正样本')

        warns.append('🔴 提醒：ERC-8004 是自报身份，99.8% 是空壳，'
                     '只能用 tier=active 子集')
        return self._emit(context, issues, warns)


# ================================================================
#  Erc8004Fetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class Erc8004Fetcher(BaseFetcher):
    """
    薄适配层：把 Erc8004DataSource 接进项目的 Fetcher 体系。

    数据逻辑全在 Erc8004DataSource；本类只负责参数透传与产出对接。
    """

    name = 'erc8004'
    desc = 'ERC-8004 注册表（须活跃度分层）'
    quality = '★★'
    needs_key = False
    fields = ['address', 'source', 'note', 'tier', 'tx_count']

    def __init__(self, chain='base', limit=200, with_uri=False,
                 no_stratify=False, **kw):
        super().__init__(**kw)
        self.chain = chain
        self.limit = int(limit or 200)
        self.with_uri = bool(with_uri)
        self.stratify = not bool(no_stratify)
        self.ds = Erc8004DataSource(chain=chain)
        self.df = pd.DataFrame()

    @property
    def output(self):
        return f'erc8004_{self.chain}.csv'

    def fetch(self) -> list:
        self.df = self.ds.fetch_identities(
            limit=self.limit, with_uri=self.with_uri,
            stratify=self.stratify, save_to_csv=True)
        self.ds.validate_identities(self.df)
        self.stats['RPC 调用次数'] = self.ds.n_calls
        self.stats['注册身份'] = len(self.df)
        if 'tier' in self.df.columns:
            self.stats['其中 active'] = int((self.df['tier'] == 'active').sum())
        return []      # 已在 fetch_identities 内落盘

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--chain', default='base',
                        choices=['base', 'ethereum', 'gnosis', 'optimism',
                                 'arbitrum', 'polygon'])
        ap.add_argument('--limit', type=int, default=200, help='最多扫多少个 agentId')
        ap.add_argument('--with-uri', action='store_true',
                        help='同时取 tokenURI（慢一倍，但分层更准）')
        ap.add_argument('--no-stratify', action='store_true', help='跳过活跃度分层')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='ERC-8004 注册表链上直读（agent 候选正样本，须分层）')
    Erc8004Fetcher.add_args(ap)
    args = ap.parse_args()
    Erc8004Fetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
