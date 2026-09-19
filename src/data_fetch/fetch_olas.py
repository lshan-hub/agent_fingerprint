#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_olas.py
@Description:
    Olas (Autonolas) ServiceRegistry 链上直读 —— agent 正样本，CSV 落盘，附 validate_* 校验。

    ============================================================
    一、 认证与限频
    ============================================================
      方式:   链上直读，无需 API Key、无需注册、零成本
      RPC:    src/utils/_base.RPC_POOL（多端点轮询 + 失败拉黑 + 指数退避）
      限频:   ⚠️ 公共 RPC 限流严格 —— 实测 mainnet.base.org 在 0.15s 间隔即 429
              RpcPool 默认 0.12s 间隔 + 3 个端点轮询，实际单端点约 0.36s

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — Olas ServiceRegistry (agent 的 Gnosis Safe 地址) ★真值最高 ───┐
    │  接口:  eth_call totalSupply()          → service 总数                    │
    │         eth_call getService(serviceId)  → 结构体，word[2] = multisig      │
    │  函数:  fetch_agents() (内部 _fetch_registry_addresses / _fetch_chain)    │
    │  覆盖:  8 条链（ethereum/base/gnosis/optimism/polygon/arbitrum/celo/mode）│
    │  更新频率: 实时（随链上区块推进）                                          │  ★必填
    │  更新时间: 无固定刷新点                                                    │  ★必填
    │  API KEY: 否 —— 链上直读，完全免费                                        │  ★必填
    │  历史范围: 当前全量快照（枚举 serviceId 1..totalSupply）                   │  ★必填
    │  产出: data/sources/olas_agents.csv                                       │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 合约地址来源: valory-xyz/autonolas-registries/docs/configuration.json（官方仓库）
        在线拉取失败时回退到内置的实测地址（FALLBACK，2026-09-03 已链上验证存在）。
      ※ 🎯 为什么它是最干净的真值: Olas 的 service 必须质押 OLAS 且实际运营才能拿激励
        ⇒ 几乎不存在「注册了但从没用过」的空壳（对比 ERC-8004 的 96% 空壳率）。
      ※ 实测（2026-09-03）: Base 636 个 service / Gnosis 3,799 个，
        提取出的 multisig 全部是 171 字节的 Gnosis Safe 代理合约。
      ※ ⚠️ Olas agent 用的是 Gnosis Safe（合约账户）而非 EOA。合约账户不能主动发交易，
        实际交易由 Safe 的 owner 或 relayer 发起 —— 特征工程上必须处理这个差异，
        并在论文中说明。

      链上原始返回          → 统一字段        → 说明
      getService().word[2] → address         agent 的 Gnosis Safe 地址（★正样本）
      getService().word[1] → (不入库)        securityDeposit 质押金额
      [遍历下标]            → note            serviceId=N
      [链名]                → source          olas_{chain}
      eth_getCode 长度      → note 附加       --verify 时标注是否为合约

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. multisig 为零地址 ⇒ service 已注册但尚未 deploy Safe，跳过
      2. 同一 Safe 可能服务多个 serviceId ⇒ 按 address 去重（保留首次出现）
      3. --verify 时用 eth_getCode 校验确实是合约（Safe 代理约 171 字节）

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🥇 三组真值中「agent 正样本」的**首选来源**，真值质量 ★★★。
        单 Olas 一个来源（Base 636 + Gnosis 3,799）就已超过现有 SOTA 全部真值
        （arXiv 2403.19530 仅 133 human + 137 bot = 270 个地址）的 16 倍。
'''

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils._base import (BaseFetcher, RpcPool, enc_uint,  # noqa: E402
                             http_json, log, selector, word_to_addr, words,
                             write_csv)


# ================================================================
#  OlasDataSource — Olas ServiceRegistry 链上直读
# ================================================================
class OlasDataSource(object):
    """
    Olas (Autonolas) ServiceRegistry 数据采集与管理

    核心函数 (共 1 个):
      ① fetch_agents(chains, limit, verify)
          eth_call totalSupply / getService → agent 的 Gnosis Safe 地址

    辅助函数:
      - registry_addresses()  在线取官方合约地址，失败回退内置
      - fetch_chain()         单链枚举

    校验函数: validate_agents

    Args:
        offline: True 时跳过在线配置拉取，直接用内置 FALLBACK 地址
    """

    # 官方配置文件（权威来源）
    CONFIG_URL = ('https://raw.githubusercontent.com/valory-xyz/'
                  'autonolas-registries/main/docs/configuration.json')

    # 回退地址（2026-09-03 实测确认，均已链上验证存在）
    FALLBACK = {
        'ethereum': ('ServiceRegistry',   '0x48b6af7B12C71f09e2fC8aF4855De4Ff54e775cA'),
        'base':     ('ServiceRegistryL2', '0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE'),
        'gnosis':   ('ServiceRegistryL2', '0x9338b5153AE39BB89f50468E608eD9d764B755fD'),
        'optimism': ('ServiceRegistryL2', '0x3d77596beb0f130a4415df3D2D8232B3d3D31e44'),
        'polygon':  ('ServiceRegistryL2', '0xE3607b00E75f6405248323A9417ff6b39B244b50'),
        'arbitrum': ('ServiceRegistryL2', '0xE3607b00E75f6405248323A9417ff6b39B244b50'),
        'celo':     ('ServiceRegistryL2', '0xE3607b00E75f6405248323A9417ff6b39B244b50'),
        'mode':     ('ServiceRegistryL2', '0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE'),
    }

    SEL_TOTAL = selector('totalSupply()')          # 0x18160ddd
    SEL_GET = selector('getService(uint256)')      # 0xef0e239b
    MULTISIG_WORD = 2   # 实测：getService 返回的第 3 个 word 是 multisig
    SAFE_CODE_LEN = 171  # Gnosis Safe 代理合约的典型字节数

    def __init__(self, offline: bool = False):
        """
        初始化

        Args:
            offline: True 时跳过在线配置，直接用内置 FALLBACK 地址
        Returns:
            None
        """
        self.offline = bool(offline)
        self.n_calls = 0
        self._registry = None

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

    @staticmethod
    def _safe_str(v, default: str = ''):
        """安全转 str: None/NaN → default。"""
        if v is None:
            return default
        s = str(v)
        return default if s.lower() in ('nan', 'none') else s

    def registry_addresses(self) -> dict:
        """
        取各链的 ServiceRegistry 合约地址

        优先在线拉官方仓库配置（保证跟随协议升级），失败回退到内置实测地址。

        Returns:
            dict: {chain: (contract_name, address)}
        """
        if self._registry is not None:
            return self._registry
        if self.offline:
            self._registry = self.FALLBACK
            return self._registry

        try:
            data = http_json(self.CONFIG_URL, timeout=25)
            nets = data if isinstance(data, list) else [data]
            out = {}
            for net in nets:
                nm = self._safe_str(net.get('name')).lower()
                nm = 'ethereum' if nm == 'mainnet' else nm
                for c in net.get('contracts', []):
                    if c.get('name') in ('ServiceRegistry', 'ServiceRegistryL2'):
                        out[nm] = (c['name'], c['address'])
            if out:
                log(f'已从官方仓库载入 {len(out)} 条链的 registry 地址', 'ok')
                self._registry = out
                return out
        except Exception as e:
            log(f'在线配置拉取失败（{str(e)[:50]}），回退内置地址', 'warn')

        self._registry = self.FALLBACK
        return self._registry

    # ============================================================
    #  ① Olas ServiceRegistry → agent 正样本
    # ============================================================
    def fetch_chain(self, chain: str, addr: str, cname: str,
                    limit: int = 0, verify: bool = False) -> list:
        """
        枚举单条链的 ServiceRegistry

        Args:
            chain:  链名（须在 _base.RPC_POOL 中有配置）
            addr:   ServiceRegistry 合约地址
            cname:  合约名（ServiceRegistry / ServiceRegistryL2）
            limit:  只取前 N 个 serviceId（0 = 全部）
            verify: 是否用 eth_getCode 校验 multisig 确实是合约
        Returns:
            list[dict]: [{address, source, note}]
        """
        pool = RpcPool(chain)
        total = self._safe_int(
            pool.call('eth_call', [{'to': addr, 'data': self.SEL_TOTAL}, 'latest']))
        n = min(total, limit) if limit else total
        log(f'[{chain}] {cname} @ {addr}', 'step')
        log(f'  totalSupply = {total:,} 个 service，本次抓取 {n:,} 个')

        rows, n_zero, n_contract = [], 0, 0
        for sid in range(1, n + 1):
            try:
                res = pool.call(
                    'eth_call',
                    [{'to': addr, 'data': self.SEL_GET + enc_uint(sid)}, 'latest'])
                w = words(res)
                if len(w) <= self.MULTISIG_WORD:
                    continue
                ms = word_to_addr(w[self.MULTISIG_WORD])
                if int(ms, 16) == 0:
                    n_zero += 1          # 1. 已注册但尚未 deploy Safe
                    continue
                row = {'address': ms.lower(), 'source': f'olas_{chain}',
                       'note': f'serviceId={sid}'}
                if verify:               # 3. 校验确实是合约
                    code = pool.call('eth_getCode', [ms, 'latest'])
                    is_c = len(code) > 2
                    n_contract += is_c
                    row['note'] += f" safe={'yes' if is_c else 'NO'} code={len(code)//2-1}B"
                rows.append(row)
            except Exception as e:
                log(f'  serviceId={sid} 失败: {str(e)[:60]}', 'err')

            if sid % 50 == 0 or sid == n:
                log(f'  进度 {sid}/{n} · 已得 {len(rows)} 个 Safe · RPC {pool.n_calls}')

        self.n_calls += pool.n_calls
        log(f'[{chain}] {len(rows)} 个 multisig，{n_zero} 个未部署'
            + (f'，{n_contract} 个确认为合约' if verify else ''), 'ok')
        return rows

    def fetch_agents(self, chains='base,gnosis', limit: int = 0,
                     verify: bool = False,
                     save_to_csv: bool = True) -> pd.DataFrame:
        """
        ① 获取 Olas agent 的 Gnosis Safe 地址（agent 正样本）

        【数据源】 eth_call totalSupply()         → service 总数
                 eth_call getService(serviceId) → 结构体，word[2] = multisig
                 - 无需 API Key，链上直读，零成本
                 - 合约地址来自官方仓库 configuration.json，失败回退内置实测地址
                 - 覆盖 8 条链；实测 Base 636 / Gnosis 3,799 个 service
                 getService 返回 11 个 32 字节 word，其中:
                    word[0] = 0x20              ← 动态结构体偏移
                    word[1] = securityDeposit
                    word[2] = multisig          ← ★ agent 的 Gnosis Safe 地址
                    word[3..] = 其余字段

        【指标含义】 一句话: "真正在运营的链上 AI agent"
                    Olas 的 service 必须质押 OLAS 且实际运营才能拿激励 ⇒
                    几乎不存在「注册了但从没用过」的空壳，是本课题真值质量最高的来源。

        【返回字段】
                  · address — agent 的 Gnosis Safe 地址（小写，★正样本）
                  · source  — olas_{chain}
                  · note    — serviceId=N [+ safe=yes/NO code=NNNB]

        【阈值解读】 无阈值。--verify 时 Safe 代理合约字节数典型为 171B；
                    显著偏离说明该 multisig 不是标准 Safe，需人工核查。

        【如何使用】
                   1) 直接作为三分类的 agent 正样本（真值质量 ★★★）
                   2) ⚠️ Olas agent 是 Gnosis Safe（合约账户）不是 EOA ——
                      合约账户不能主动发交易，实际交易由 owner/relayer 发起。
                      特征工程必须处理这个差异，并在论文中说明
                   3) 与 x402 付款方（弱标签）分层使用：Olas 作高置信集，
                      x402 作扩充集，训练时用 PU learning 处理标签噪声

        Args:
            chains:      逗号分隔的链名，默认 'base,gnosis'
            limit:       每链只取前 N 个 serviceId（0 = 全部）
            verify:      是否用 eth_getCode 校验 multisig 是合约
            save_to_csv: 是否写入 data/sources/olas_agents.csv
        Returns:
            pd.DataFrame: [address, source, note]
        """
        chain_list = [c.strip() for c in str(chains).split(',') if c.strip()]
        reg = self.registry_addresses()

        all_rows = []
        for chain in chain_list:
            if chain not in reg:
                log(f'[{chain}] 无 registry 配置，跳过', 'warn')
                continue
            cname, addr = reg[chain]
            try:
                all_rows += self.fetch_chain(chain, addr, cname, limit, verify)
            except Exception as e:
                log(f'[{chain}] 整链失败: {e}', 'err')

        if not all_rows:
            log('没有抓到任何地址', 'warn')
            return pd.DataFrame()

        # 2. 同一 Safe 可能服务多个 serviceId ⇒ 去重
        df = (pd.DataFrame(all_rows)
              .drop_duplicates(subset=['address'], keep='first')
              .reset_index(drop=True))
        log(f'去重: {len(all_rows)} → {len(df)} 个唯一 agent 地址', 'ok')

        if save_to_csv:
            write_csv('olas_agents.csv', df.to_dict('records'), list(df.columns),
                      note=('Olas ServiceRegistry 的 multisig(Gnosis Safe) 地址\n'
                            'agent 正样本，真值质量 ★★★（须质押+实际运营，几乎无空壳）\n'
                            '⚠️ 是合约账户(Safe)不是 EOA，特征工程需处理此差异\n'
                            f'链: {chains} | 去重前 {len(all_rows)} → 后 {len(df)}'))
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

    def validate_agents(self, df: pd.DataFrame,
                        context: str = 'Olas agent') -> bool:
        """
        校验 ① 的产出

        检查项:
          1. 非空
          2. 必需列齐全
          3. 地址格式合法且无重复
          4. 零地址不应出现
          5. ⚠️ --verify 时: 非合约地址告警（应全部是 Safe）
          6. ⚠️ 样本量是否够训练

        Args:
            df:      fetch_agents 的结果
            context: 日志上下文名
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
            issues.append(f'{dup} 个地址重复（去重逻辑有误）')
        zero = int((df['address'] == '0x' + '0' * 40).sum())
        if zero:
            issues.append(f'{zero} 个零地址（应在抓取时已过滤）')

        if df['note'].astype(str).str.contains('safe=').any():
            n_no = int(df['note'].astype(str).str.contains('safe=NO').sum())
            if n_no:
                warns.append(f'{n_no} 个 multisig 不是合约 —— 可能是未部署的 Safe，'
                             f'建议剔除后再作正样本')

        if len(df) < 100:
            warns.append(f'仅 {len(df)} 个地址，样本偏少。'
                         f'建议 --chains base,gnosis 全量抓取（实测可得 4,400+）')

        warns.append('🔴 提醒：Olas agent 是 Gnosis Safe（合约账户）不是 EOA，'
                     '特征工程需处理此差异')
        return self._emit(context, issues, warns)


# ================================================================
#  OlasFetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class OlasFetcher(BaseFetcher):
    """
    薄适配层：把 OlasDataSource 接进项目的 Fetcher 体系。

    数据逻辑全在 OlasDataSource；本类只负责参数透传与产出对接。
    """

    name = 'olas'
    desc = 'Olas agent Safe 地址（真值质量最高）'
    output = 'olas_agents.csv'
    quality = '★★★'
    needs_key = False
    fields = ['address', 'source', 'note']

    def __init__(self, chains='base,gnosis', limit=0, verify=False,
                 offline=False, **kw):
        super().__init__(**kw)
        self.chains = chains
        self.limit = int(limit or 0)
        self.verify = bool(verify)
        self.ds = OlasDataSource(offline=offline)
        self.df = pd.DataFrame()

    def fetch(self) -> list:
        self.df = self.ds.fetch_agents(chains=self.chains, limit=self.limit,
                                       verify=self.verify, save_to_csv=True)
        self.ds.validate_agents(self.df)
        self.stats['RPC 调用次数'] = self.ds.n_calls
        self.stats['唯一 agent 地址'] = len(self.df)
        return []      # 已在 fetch_agents 内落盘

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--chains', default='base,gnosis',
                        help='逗号分隔，可选：' + ','.join(OlasDataSource.FALLBACK))
        ap.add_argument('--limit', type=int, default=0, help='每链只取前 N 个（0=全部）')
        ap.add_argument('--verify', action='store_true',
                        help='用 eth_getCode 校验 multisig 是合约')
        ap.add_argument('--offline', action='store_true', help='跳过在线配置拉取')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Olas ServiceRegistry 链上直读（agent 正样本）')
    OlasFetcher.add_args(ap)
    args = ap.parse_args()
    OlasFetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
