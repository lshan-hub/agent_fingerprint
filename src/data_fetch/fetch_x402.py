#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_x402.py
@Description:
    x402 付款方提取（SQD 日志扫描）—— agent 弱正样本 + Facilitator 副产品，附 validate_* 校验。

    ============================================================
    一、 认证与限频
    ============================================================
      方式:   SQD Network 公开 Portal，无需 API Key、无需注册、零成本
      Base:   https://portal.sqd.dev/datasets/base-mainnet/stream
      限频:   无硬限流，但单次响应只推进约 396 区块（SQDClient.stream 已自动分页）
      ⚠️ 必须设 User-Agent，否则 403（SQDClient 已内置）

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — x402 付款方 (EIP-3009 元交易的 authorizer) ★规模最大 ─────────┐
    │  接口:  SQD stream，过滤 Base USDC 的两类日志                             │
    │           AuthorizationUsed(address,bytes32)  ← ★定位付款人               │
    │           Transfer(address,address,uint256)   ← 提供收款人与金额          │
    │  函数:  fetch_payers() (内部 _build_query / _scan)                        │
    │  覆盖:  Base 链 USDC                                                      │
    │  更新频率: 实时（随链上区块推进）                                          │  ★必填
    │  更新时间: 无固定刷新点                                                    │  ★必填
    │  API KEY: 否 —— SQD 公开 Portal，完全免费                                 │  ★必填
    │  历史范围: ⭐ 全历史（SQD start_block=0）；默认从链头往回扫 --blocks 个    │  ★必填
    │  产出: data/sources/x402_payers.csv + x402_facilitators.csv               │
    └───────────────────────────────────────────────────────────────────────────┘
      Base USDC 合约（已链上验证 symbol()="USDC" decimals()=6）:
        0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913

      ※ 🔑 核心难点 —— 付款人 ≠ 交易发起人:
        x402 用 EIP-3009 元交易，agent 只签名，Facilitator 代付 gas 并提交上链。
        ⇒ 链上 tx.from 是 Facilitator，真正的 agent 藏在授权参数里。
        ⚠️ 只看 tx.from 会把成千上万个 agent 全部误认成同一个 Facilitator 地址。

      ※ ⚠️ 走过的弯路（保留作教训）:
        最初用「Transfer.from != tx.from」判断代付，实测 **91.1%** 的 USDC 转账都满足
        —— 因为经 DEX router / 聚合器中转的转账也满足这个条件。
        改用 AuthorizationUsed 后占比降到 **4.7%**，才是 EIP-3009 的真实比例。

      链上原始返回              → 统一字段      → 说明
      AuthorizationUsed.topic1 → address       ★真正的付款人（agent）
      Transfer.topic2          → (聚合)        收款人，计入 n_sellers
      Transfer.data            → avg_usd       金额（USDC 6 位小数）
      tx.from                  → (副产品)      Facilitator，单独输出
      [按地址计数]              → n_payments    支付笔数

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. 只统计 AuthorizationUsed 事件（EIP-3009 专属，唯一可靠的元交易信号）
      2. 同一笔交易内配对 Transfer（from == payer）取收款人与金额
      3. 提纯门槛:
         · n_payments >= min_payments  —— 过滤偶发的人工调用
         · avg_usd    <= max_avg_usd   —— x402 均单 $0.31，人类不会手工发数千笔微支付
      4. 健全性检查: 元交易占 Transfer 比例 > 40% 即告警（过滤器可能又写错了）

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🥈 agent 正样本的**规模最大来源**，真值质量 ★★（弱标签）。
        Base 上 x402 累计 1.19 亿笔交易、约 19 万买家/30 天。
        与 Olas（★★★ 高置信但量小）分层使用，训练时用 PU learning 处理噪声。
      💡 副产品 Facilitator 清单同样有价值 —— 它标识了 x402 的基础设施方，
        且这些地址必须从 agent 样本中排除（它们是代付网关，不是 agent）。
'''

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils._base import BaseFetcher, log, topic0, write_csv  # noqa: E402
from src.utils.sqd_client import SQDClient  # noqa: E402


# ================================================================
#  X402DataSource — x402 付款方与 Facilitator 提取
# ================================================================
class X402DataSource(object):
    """
    x402 (EIP-3009 元交易) 数据采集与管理

    核心函数 (共 1 个):
      ① fetch_payers(blocks, from_block, min_payments, max_avg_usd)
          扫 USDC 的 AuthorizationUsed + Transfer 日志 → 付款方(agent) + Facilitator

    辅助函数:
      - _build_query()  构造 SQD 查询体
      - _scan()         扫描并聚合

    校验函数: validate_payers / validate_facilitators

    Args:
        client: 可选的 SQDClient 实例；缺省时自建
    """

    USDC_BASE = '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913'
    USDC_DECIMALS = 6

    TOPIC_TRANSFER = topic0('Transfer(address,address,uint256)')
    TOPIC_AUTH = topic0('AuthorizationUsed(address,bytes32)')   # ★ EIP-3009 专属

    # 元交易占 Transfer 的比例超过这个值就说明过滤器有问题（实测正常约 4.7%）
    SANITY_MAX_RATIO = 40.0
    # x402 官方口径的平均单笔金额，用于校验器判断样本是否像 x402
    X402_TYPICAL_AVG_USD = 0.31

    def __init__(self, client: SQDClient = None):
        """
        初始化

        Args:
            client: 可选的 SQDClient；缺省自建（读 config.SQD_DATASET）
        Returns:
            None
        """
        self.client = client or SQDClient()
        self.payers = {}
        self.facilitators = {}
        self.n_auth = 0
        self.n_transfer = 0

    # ============================================================
    #  通用工具
    # ============================================================
    @staticmethod
    def _topic_to_addr(t: str) -> str:
        """32 字节 topic → 20 字节地址（小写）。"""
        return ('0x' + t[-40:]).lower()

    @staticmethod
    def _safe_float(v, default: float = 0.0):
        """安全转 float（支持 0x 前缀十六进制）。"""
        try:
            if v is None or v == '':
                return default
            if isinstance(v, str) and v.startswith('0x'):
                return float(int(v, 16))
            fv = float(v)
            return default if fv != fv else fv
        except (TypeError, ValueError):
            return default

    def _build_query(self, lo: int, hi: int) -> dict:
        """
        构造 SQD 查询体

        同时拉两种事件: AuthorizationUsed 定位付款人，Transfer 提供金额与收款人。

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
                'log': {'address': True, 'topics': True, 'data': True,
                        'transactionIndex': True, 'logIndex': True},
                'transaction': {'from': True, 'transactionIndex': True},
            },
            'logs': [
                {'address': [self.USDC_BASE], 'topic0': [self.TOPIC_AUTH],
                 'transaction': True},
                {'address': [self.USDC_BASE], 'topic0': [self.TOPIC_TRANSFER]},
            ],
        }

    def _scan(self, lo: int, hi: int) -> tuple:
        """
        扫描区块范围，聚合付款方与 Facilitator

        AuthorizationUsed 只在 transferWithAuthorization / receiveWithAuthorization
        成功时触发 —— 这是识别元交易的唯一可靠信号（清洗逻辑 1）。

        Args:
            lo / hi: 区块范围
        Returns:
            (payers: dict, facilitators: dict)
        """
        payers = defaultdict(lambda: {'n': 0, 'usd': 0.0, 'sellers': set(),
                                      'facilitators': set(),
                                      'first': None, 'last': None})
        facils = defaultdict(int)
        n_auth = n_tf = 0

        for blk in self.client.stream(self._build_query(lo, hi)):
            ts = blk.get('header', {}).get('timestamp')
            tx_from = {t.get('transactionIndex'): (t.get('from') or '').lower()
                       for t in blk.get('transactions', [])}

            transfers, auths = defaultdict(list), []
            for lg in blk.get('logs', []):
                tp = lg.get('topics') or []
                if not tp:
                    continue
                if tp[0] == self.TOPIC_AUTH and len(tp) >= 2:
                    auths.append(lg)
                elif tp[0] == self.TOPIC_TRANSFER and len(tp) >= 3:
                    transfers[lg.get('transactionIndex')].append(lg)
                    n_tf += 1

            for lg in auths:
                n_auth += 1
                txi = lg.get('transactionIndex')
                payer = self._topic_to_addr(lg['topics'][1])      # ★ authorizer
                sender = tx_from.get(txi, '')

                # 2. 同一笔交易内配对 Transfer
                amt, to = 0.0, ''
                for tf in transfers.get(txi, []):
                    if self._topic_to_addr(tf['topics'][1]) == payer:
                        to = self._topic_to_addr(tf['topics'][2])
                        amt = self._safe_float(tf.get('data')) / (10 ** self.USDC_DECIMALS)
                        break

                p = payers[payer]
                p['n'] += 1
                p['usd'] += amt
                if to:
                    p['sellers'].add(to)
                if sender:
                    p['facilitators'].add(sender)
                    facils[sender] += 1
                if ts:
                    p['first'] = ts if p['first'] is None else min(p['first'], ts)
                    p['last'] = ts if p['last'] is None else max(p['last'], ts)

        self.n_auth, self.n_transfer = n_auth, n_tf
        ratio = n_auth / max(n_tf, 1) * 100
        log(f'扫描完成：{n_auth:,} 条 AuthorizationUsed（EIP-3009 元交易）'
            f' / {n_tf:,} 条 Transfer，占比 {ratio:.1f}%', 'ok')
        if ratio > self.SANITY_MAX_RATIO:      # 4. 健全性检查
            log('  ⚠️ 元交易占比异常高，请检查过滤器是否正确', 'warn')
        return payers, facils

    # ============================================================
    #  ① x402 付款方 → agent 弱正样本
    # ============================================================
    def fetch_payers(self, blocks: int = 20000, from_block: int = 0,
                     min_payments: int = 5, max_avg_usd: float = 5.0,
                     save_to_csv: bool = True) -> pd.DataFrame:
        """
        ① 从 EIP-3009 元交易提取 x402 付款方（agent 弱正样本）

        【数据源】 SQD stream，过滤 Base USDC 的两类日志
                 - AuthorizationUsed(address,bytes32) ← ★ topic1 = 真正的付款人
                 - Transfer(address,address,uint256)  ← 提供收款人与金额
                 - 无需 API Key，SQD 公开 Portal 免费
                 - 单次响应只推进约 396 区块，SQDClient.stream 已自动分页

        【指标含义】 一句话: "为 API 调用自动付费的链上地址"
                    x402 平均单笔 $0.31 —— 人类不可能手工发起数千笔微支付。
                    🔑 但付款人 ≠ 交易发起人: Facilitator 代付 gas，
                       链上 tx.from 是 Facilitator，agent 藏在 EIP-3009 授权参数里。

        【返回字段】
                  · address     — 付款方地址（★agent 候选）
                  · source      — x402_payer
                  · note        — n=N avg=$X sellers=N facilitators=N
                  · n_payments  — 支付笔数
                  · avg_usd     — 平均单笔金额
                  · n_sellers   — 交互过的收款方数量

        【阈值解读】
                    · n_payments < 5      → 偶发调用，可能是人工，剔除
                    · avg_usd > $5        → 不像 x402 微支付，剔除
                    · x402 官方口径均单 $0.31 —— 样本均值应在此量级
                    · 元交易占 Transfer 比例正常约 4.7%，>40% 说明过滤器有误

        【如何使用】
                   1) 作为 agent 正样本的**扩充集**（量大但是弱标签）
                   2) 与 Olas（★★★ 高置信）分层：Olas 作高置信集，
                      x402 作扩充集，训练时用 PU learning / confident learning
                   3) 🔴 Facilitator 必须从 agent 样本中排除 ——
                      它们是代付网关，不是 agent。调 save_facilitators() 取清单
                   4) 提高 min_payments 到 20 可显著提纯（正式跑建议值）

        Args:
            blocks:       从链头往回扫多少区块（Base 2s/块）
            from_block:   指定起始区块（覆盖 blocks）
            min_payments: 单地址最少支付笔数
            max_avg_usd:  均单金额上限
            save_to_csv:  是否写入 data/sources/x402_payers.csv
        Returns:
            pd.DataFrame
        """
        head = self.client.head()
        hi = head
        lo = from_block or max(0, head - blocks)

        log('① 扫 Base USDC 的 EIP-3009 日志', 'step')
        log(f'  合约 {self.USDC_BASE}')
        log(f'  区块 {lo:,} → {hi:,}（{hi-lo:,} 块）')

        self.payers, self.facilitators = self._scan(lo, hi)
        if not self.payers:
            log('未发现元交易。可能该窗口内无 x402 活动，试试加大 blocks', 'warn')
            return pd.DataFrame()

        rows = []
        for addr, s in self.payers.items():
            avg = s['usd'] / max(s['n'], 1)
            if s['n'] < min_payments or avg > max_avg_usd:    # 3. 提纯门槛
                continue
            rows.append({
                'address': addr, 'source': 'x402_payer',
                'note': (f"n={s['n']} avg=${avg:.4f} sellers={len(s['sellers'])} "
                         f"facilitators={len(s['facilitators'])}"),
                'n_payments': s['n'], 'avg_usd': round(avg, 6),
                'n_sellers': len(s['sellers']),
            })
        if not rows:
            log(f'过滤后无剩余（门槛 n>={min_payments} avg<=${max_avg_usd}）', 'warn')
            return pd.DataFrame()

        df = (pd.DataFrame(rows)
              .sort_values('n_payments', ascending=False)
              .reset_index(drop=True))
        log(f'  付款方 {len(self.payers):,} 个 → 过滤后 {len(df):,} 个', 'ok')
        log(f'  识别出 {len(self.facilitators)} 个 Facilitator', 'ok')

        if save_to_csv:
            write_csv('x402_payers.csv', df.to_dict('records'), list(df.columns),
                      note=('x402 付款方（agent 弱正样本，真值质量 ★★）\n'
                            '★ 判据：EIP-3009 的 AuthorizationUsed 事件，'
                            'topic1 = 真正的付款人\n'
                            '  切勿用 tx.from —— 那是代付 gas 的 Facilitator\n'
                            f'区块 {lo}-{hi} | 门槛 n>={min_payments} '
                            f'avg<=${max_avg_usd}'))
            self.save_facilitators()
        return df

    # ============================================================
    #  ② Facilitator 画像（30k 方案的提纯依据）
    # ============================================================
    def profile_facilitators(self, lo: int, hi: int,
                             slices: int = None,
                             slice_blocks: int = None) -> pd.DataFrame:
        """
        ② 按「提交者」聚合的 facilitator 画像 —— 30k 方案的 x402 提纯依据

        【为什么需要】 EIP-3009 不止 x402 在用：Coinbase 智能钱包的免 gas
                    转账同样触发 AuthorizationUsed。全窗口 auth-only 扫描
                    （sweep_window main 趟）拿不到金额，无法按 avg_usd 提纯。
                    但提交者（tx.from）只有几十个 —— 在窗口内采样若干切片、
                    配对 Transfer 金额，就能给每个提交者画像:
                      x402 facilitator: 海量小额（官方口径均单 $0.31）
                      零售中继:         大额转账
                    装配端按「付款方经零售型提交者的支付占比」剔除污染。

        【产出】 x402_facilitators.csv 增加 avg_usd / n_payers 列，
                供 sweep_window assemble 读取分类。

        Args:
            lo / hi:      建模窗口
            slices:       采样切片数（缺省 config.X402_PROFILE_SLICES)
            slice_blocks: 每切片块数（缺省 config.X402_PROFILE_SLICE_BLOCKS)
        Returns:
            pd.DataFrame
        """
        from conf import config as _c
        slices = slices or _c.X402_PROFILE_SLICES
        slice_blocks = slice_blocks or _c.X402_PROFILE_SLICE_BLOCKS
        span = max(hi - lo, 1)
        step = max((span - slice_blocks) // max(slices - 1, 1), 1)

        subs = defaultdict(lambda: {'n': 0, 'usd': 0.0, 'payers': set()})
        for i in range(slices):
            s_lo = min(lo + i * step, hi - 1)
            s_hi = min(s_lo + slice_blocks, hi)
            log(f'  [profile] 切片 {i+1}/{slices} 区块 {s_lo:,}→{s_hi:,}')
            for blk in self.client.stream(self._build_query(s_lo, s_hi)):
                tx_from = {t.get('transactionIndex'): (t.get('from') or '').lower()
                           for t in blk.get('transactions', [])}
                transfers, auths = defaultdict(list), []
                for lg in blk.get('logs', []):
                    tp = lg.get('topics') or []
                    if not tp:
                        continue
                    if tp[0] == self.TOPIC_AUTH and len(tp) >= 2:
                        auths.append(lg)
                    elif tp[0] == self.TOPIC_TRANSFER and len(tp) >= 3:
                        transfers[lg.get('transactionIndex')].append(lg)
                for lg in auths:
                    txi = lg.get('transactionIndex')
                    payer = self._topic_to_addr(lg['topics'][1])
                    sender = tx_from.get(txi, '')
                    if not sender:
                        continue
                    amt = 0.0
                    for tf in transfers.get(txi, []):
                        if self._topic_to_addr(tf['topics'][1]) == payer:
                            amt = (self._safe_float(tf.get('data'))
                                   / (10 ** self.USDC_DECIMALS))
                            break
                    s = subs[sender]
                    s['n'] += 1
                    s['usd'] += amt
                    s['payers'].add(payer)

        rows = []
        for a, s in sorted(subs.items(), key=lambda x: -x[1]['n']):
            avg = s['usd'] / max(s['n'], 1)
            kind = ('x402' if avg <= _c.X402_FACIL_GOOD_MAX_AVG
                    else ('retail' if avg >= _c.X402_FACIL_BAD_MIN_AVG
                          else 'mixed'))
            rows.append({'address': a, 'source': 'x402_facilitator',
                         'note': f'settled={s["n"]} avg=${avg:.4f} '
                                 f'payers={len(s["payers"])} kind={kind}',
                         'n_settled': s['n'], 'avg_usd': round(avg, 6),
                         'n_payers': len(s['payers']), 'kind': kind})
        df = pd.DataFrame(rows)
        if not df.empty:
            write_csv('x402_facilitators.csv', df.to_dict('records'),
                      list(df.columns),
                      note=('x402 Facilitator/提交者画像（采样切片配对金额）\n'
                            'kind=x402: 均单 ≤$5（微支付网关）\n'
                            'kind=retail: 均单 ≥$20（零售中继，'
                            '其付款方大概率不是 agent）\n'
                            '🔴 这些地址必须从 agent 样本中排除；'
                            'retail 型用于装配端剔除被污染的付款方\n'
                            f'窗口 {lo}-{hi} | {slices} 切片 × {slice_blocks} 块'))
            n_kind = df['kind'].value_counts().to_dict()
            log(f'  [profile] 提交者 {len(df)} 个: {n_kind}', 'ok')
        return df

    def save_facilitators(self, top: int = 20) -> pd.DataFrame:
        """
        输出 Facilitator 清单（副产品）

        Facilitator 是代付 gas 的基础设施方，由高频代付行为反推识别。
        🔴 这些地址必须从 agent 样本中排除。

        Args:
            top: 只输出代付笔数最高的前 N 个
        Returns:
            pd.DataFrame
        """
        if not self.facilitators:
            return pd.DataFrame()
        rank = sorted(self.facilitators.items(), key=lambda x: -x[1])[:top]
        df = pd.DataFrame([{'address': a, 'source': 'x402_facilitator',
                            'note': f'settled={n}', 'n_settled': n}
                           for a, n in rank])
        write_csv('x402_facilitators.csv', df.to_dict('records'), list(df.columns),
                  note=('x402 Facilitator（代付 gas 的基础设施方）\n'
                        '由高频代付行为反推识别\n'
                        '🔴 这些地址必须从 agent 样本中排除'))
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

    def validate_payers(self, df: pd.DataFrame,
                        context: str = 'x402 付款方') -> bool:
        """
        校验 ① 的产出

        检查项:
          1. 非空 / 必需列齐全
          2. 地址格式合法且无重复
          3. n_payments / avg_usd 取值合理
          4. ⚠️ 元交易占比是否落在合理区间（过滤器健全性）
          5. ⚠️ 均单金额是否接近 x402 官方口径 $0.31
          6. 🔴 提示: Facilitator 必须排除

        Args:
            df / context
        Returns:
            bool
        """
        import re
        issues, warns = [], []
        addr_re = re.compile(r'^0x[0-9a-f]{40}$')
        need = ['address', 'source', 'note', 'n_payments', 'avg_usd', 'n_sellers']

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
        if (df['n_payments'] < 1).any():
            issues.append('存在支付笔数 < 1 的行')
        if (df['avg_usd'] < 0).any():
            issues.append('存在负均单金额')

        # 4. 过滤器健全性
        if self.n_transfer:
            ratio = self.n_auth / self.n_transfer * 100
            if ratio > self.SANITY_MAX_RATIO:
                issues.append(f'元交易占比 {ratio:.1f}% 异常高（正常约 4.7%）'
                              f'—— 过滤器很可能写错了')
            elif ratio < 0.5:
                warns.append(f'元交易占比仅 {ratio:.2f}%，样本可能过少，'
                             f'建议加大扫描区块数')

        # 5. 均单金额对标官方口径
        med = float(df['avg_usd'].median())
        if med > self.X402_TYPICAL_AVG_USD * 20:
            warns.append(f'均单中位数 ${med:.4f} 远高于 x402 官方口径 '
                         f'${self.X402_TYPICAL_AVG_USD}，样本可能混入非 x402 场景')

        if len(df) < 50:
            warns.append(f'仅 {len(df)} 个地址，建议加大 blocks 或降低 min_payments')

        warns.append('🔴 提醒：Facilitator 是代付网关不是 agent，'
                     '必须从样本中排除（见 x402_facilitators.csv）')
        warns.append('🔴 提醒：x402 是弱标签，训练时需用 PU learning 处理噪声')
        return self._emit(context, issues, warns)

    def validate_facilitators(self, df: pd.DataFrame,
                              context: str = 'x402 Facilitator') -> bool:
        """校验 Facilitator 副产品: 非空 + 代付笔数递减。"""
        issues, warns = [], []
        if df is None or df.empty:
            warns.append('未识别出 Facilitator（窗口内可能无 x402 活动）')
            return self._emit(context, issues, warns)
        if 'n_settled' in df.columns and not df['n_settled'].is_monotonic_decreasing:
            warns.append('未按代付笔数降序，排序逻辑可能有误')
        return self._emit(context, issues, warns)


# ================================================================
#  X402Fetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class X402Fetcher(BaseFetcher):
    """
    薄适配层：把 X402DataSource 接进项目的 Fetcher 体系。

    数据逻辑全在 X402DataSource；本类只负责参数透传与产出对接。
    """

    name = 'x402'
    desc = 'x402 付款方（agent 弱正样本）'
    output = 'x402_payers.csv'
    quality = '★★'
    needs_key = False
    fields = ['address', 'source', 'note', 'n_payments', 'avg_usd', 'n_sellers']

    def __init__(self, blocks=20000, from_block=0, min_payments=5,
                 max_avg_usd=5.0, profile=False, lo=None, hi=None, **kw):
        super().__init__(**kw)
        self.blocks = int(blocks or 20000)
        self.from_block = int(from_block or 0)
        self.min_payments = int(min_payments or 5)
        self.max_avg_usd = float(max_avg_usd or 5.0)
        self.profile = bool(profile)
        self.lo, self.hi = lo, hi
        self.ds = X402DataSource()
        self.df = pd.DataFrame()

    def fetch(self) -> list:
        if self.profile:
            # 30k 方案: 只做 facilitator 画像（供 sweep_window assemble 提纯）
            from conf import config as _c
            lo = self.lo or _c.MODEL_WINDOW_START_BLOCK
            hi = self.hi or _c.MODEL_WINDOW_END_BLOCK
            self.df = self.ds.profile_facilitators(lo, hi)
            self.stats['提交者画像'] = len(self.df)
            return []
        self.df = self.ds.fetch_payers(
            blocks=self.blocks, from_block=self.from_block,
            min_payments=self.min_payments, max_avg_usd=self.max_avg_usd,
            save_to_csv=True)
        self.ds.validate_payers(self.df)
        self.stats['元交易/Transfer'] = f'{self.ds.n_auth:,} / {self.ds.n_transfer:,}'
        self.stats['付款方（过滤后）'] = len(self.df)
        self.stats['Facilitator'] = len(self.ds.facilitators)
        return []      # 已在 fetch_payers 内落盘

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--blocks', type=int, default=20000,
                        help='从链头往回扫多少区块（Base 2s/块，20000≈11 小时）')
        ap.add_argument('--from-block', type=int, default=0,
                        help='指定起始区块（覆盖 --blocks）')
        ap.add_argument('--min-payments', type=int, default=5,
                        help='单地址最少支付笔数（提纯，正式跑建议 20）')
        ap.add_argument('--max-avg-usd', type=float, default=5.0,
                        help='均单金额上限（x402 官方口径 $0.31）')
        ap.add_argument('--profile', action='store_true',
                        help='30k 方案: 只做 facilitator 画像'
                             '（建模窗口内采样切片，配对金额分类提交者）')
        ap.add_argument('--lo', type=int, default=None,
                        help='画像窗口起点（缺省建模窗口）')
        ap.add_argument('--hi', type=int, default=None,
                        help='画像窗口终点（缺省建模窗口）')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='x402 付款方提取（EIP-3009 元交易，agent 弱正样本）')
    X402Fetcher.add_args(ap)
    args = ap.parse_args()
    X402Fetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
