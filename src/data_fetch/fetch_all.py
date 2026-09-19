#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_all.py
@Description:
    数据采集编排器 —— 一键跑通所有免费数据源，并汇总成三组真值地址清单。

    ============================================================
    一、 编排的数据源
    ============================================================
    免费源（本编排器默认全跑，零成本、无需任何 Key）:
      ① OlasFetcher          agent 正样本  ★★★  链上直读 ServiceRegistry
      ② Erc8004Fetcher       agent 正样本  ★★   链上直读注册表（须分层）
      ③ X402Fetcher          agent 正样本  ★★   SQD 扫 EIP-3009 日志
      ④ MevBotFetcher        bot 负样本    ★★★  SQD 扫原子套利
      ⑤ AgentEconomyFetcher  趋势验证      —     公开 JSON + Dune 查询 ID

    需要 Key 的源（本编排器跳过，请单独跑）:
      · TxHistoryFetcher  → src/data_fetch/fetch_txs.py       （etherscan 后端需 Key）
      · DuneFetcher       → src/data_fetch/fetch_dune.py      （需 DUNE_API_KEY）
      · CoinglassFetcher  → src/data_fetch/fetch_coinglass.py （需付费 Key）

    ⚠️ 人类样本本编排器不产出 —— 它没有免费的单一来源，需要多信号交叉
      （CEX 提现 + ENS/Farcaster + 昼夜节律，至少命中 2 条）。
      见 data/addresses/README.md。

    ============================================================
    二、 汇总逻辑（merge_groups）
    ============================================================
      agent 正样本 = olas_agents.csv
                   + erc8004_base.csv 中 tier=active 的子集   ← 🔴 dormant 必须排除
                   + x402_payers.csv
      bot 负样本   = mev_bots.csv

      1. 按 address 去重（跨源，保留首次出现）
      2. 🔴 冲突处理: 一个地址不能既是 agent 又是 bot ——
         同时命中两组判据的地址**从两组中都剔除**，需人工裁决
      3. 写入 data/addresses/{agent_olas,bot_mev}.csv

    ============================================================
    三、 两种模式
    ============================================================
      快速验证（默认）  约 5 分钟：小样本跑通全链路，验证管线可用
      全量（--full）    约 30 分钟：Olas 全枚举 + 更大区块窗口 + 更严提纯门槛

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🎬 **入口** —— 第一周只需要跑这一条命令，就能得到三组真值中的两组，
        然后接 src/feature/compute_features.py 算特征、
        src/modeling/plot_gonogo.py 出 go/no-go 判定。
'''

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils._base import OUT_DIR, log, read_source_csv  # noqa: E402

from src.data_fetch.fetch_agenteconomy import AgentEconomyFetcher  # noqa: E402
from src.data_fetch.fetch_erc8004 import Erc8004Fetcher  # noqa: E402
from src.data_fetch.fetch_mev_bots import MevBotFetcher  # noqa: E402
from src.data_fetch.fetch_olas import OlasFetcher  # noqa: E402
from src.data_fetch.fetch_x402 import X402Fetcher  # noqa: E402


# ================================================================
#  FetchOrchestrator — 数据采集编排与真值汇总
# ================================================================
class FetchOrchestrator(object):
    """
    按顺序跑各 Fetcher，再把产出汇总成 compute_features 需要的三组清单。

    核心函数 (共 3 个):
      ① fetch_all()     依次跑 JOBS 里的所有免费 Fetcher
      ② merge_groups()  跨源去重 + 冲突剔除 → 写 data/addresses/
      ③ run()           ①+②+汇报

    辅助函数:
      - run_one()  跑单个 Fetcher（异常隔离，单源失败不中断整批）
      - report()   汇报各源结果

    校验函数: validate_merged

    Args:
        full: True 走全量模式（慢，约 30 分钟）
        skip: 逗号分隔的源名，跳过它们
    """

    # (Fetcher 类, 快速模式参数, 全量模式参数)
    JOBS = [
        (OlasFetcher,
         dict(chains='base', limit=60),
         dict(chains='base,gnosis')),
        (Erc8004Fetcher,
         dict(chain='base', limit=60),
         dict(chain='base', limit=500, with_uri=True)),
        (X402Fetcher,
         dict(blocks=5000, min_payments=5),
         dict(blocks=100000, min_payments=20)),
        (MevBotFetcher,
         dict(blocks=5000, min_arb=5),
         dict(blocks=50000, min_arb=20)),
        (AgentEconomyFetcher, dict(), dict()),
    ]

    # agent 正样本的来源文件与标签
    AGENT_SOURCES = [('olas_agents.csv', 'olas'),
                     ('erc8004_base.csv', 'erc8004'),
                     ('x402_payers.csv', 'x402')]
    BOT_SOURCES = [('mev_bots.csv', 'mev')]

    def __init__(self, full: bool = False, skip: str = ''):
        """
        初始化

        Args:
            full: 是否走全量模式
            skip: 逗号分隔的源名
        Returns:
            None
        """
        self.full = bool(full)
        self.skip = {s.strip() for s in str(skip).split(',') if s.strip()}
        self.results = {}
        self.instances = {}
        self.merged = {}

    # ============================================================
    #  通用工具
    # ============================================================
    @staticmethod
    def _is_addr(a) -> bool:
        """合法 EVM 地址判定。"""
        a = str(a or '').lower()
        return a.startswith('0x') and len(a) == 42

    # ============================================================
    #  ① 依次跑各 Fetcher
    # ============================================================
    def run_one(self, cls, opts: dict) -> bool:
        """
        跑单个 Fetcher（异常隔离）

        Args:
            cls:  Fetcher 类
            opts: 构造参数
        Returns:
            bool: 是否成功
        """
        log('=' * 60, 'info')
        log(f'[{cls.name}] {cls.desc}', 'step')
        log('=' * 60, 'info')
        t0 = time.time()
        try:
            inst = cls(**opts)
            self.instances[cls.name] = inst
            inst.run()
            ok = True
        except SystemExit as e:      # 缺 key 之类的显式退出
            log(f'[{cls.name}] 跳过：{e}', 'warn')
            ok = False
        except Exception as e:
            log(f'[{cls.name}] 失败：{type(e).__name__}: {str(e)[:120]}', 'err')
            ok = False
        log(f"[{cls.name}] {'完成' if ok else '失败'}（{time.time()-t0:.0f}s）",
            'ok' if ok else 'err')
        return ok

    def fetch_all(self) -> dict:
        """
        ① 依次跑 JOBS 里的所有免费 Fetcher

        【数据源】 见模块文档 一（5 个免费源）

        【如何使用】 单源失败不中断整批，最后由 report() 汇总成败。

        Returns:
            dict: {source_name: ok}
        """
        log(f"模式: {'全量' if self.full else '快速验证'}"
            + (f' | 跳过 {self.skip}' if self.skip else ''), 'info')
        log(f'产出目录: {OUT_DIR}', 'info')
        for cls, quick, full in self.JOBS:
            if cls.name in self.skip:
                log(f'[{cls.name}] 跳过', 'warn')
                continue
            self.results[cls.name] = self.run_one(cls, full if self.full else quick)
        return self.results

    # ============================================================
    #  ② 汇总为三组真值地址
    # ============================================================
    def merge_groups(self) -> dict:
        """
        ② 把各源产出汇总成 compute_features 需要的三组地址清单

        【数据源】 data/sources/ 下各 Fetcher 的 CSV 产出

        【指标含义】 一句话: "把多源候选合并成可直接训练的真值清单"

        【返回字段】 {'agent': [...], 'bot': [...]}，每项 {address, source, note}

        【阈值解读】 🔴 ERC-8004 只收 tier=active，dormant 是空壳（96%）必须排除

        【如何使用】
                   1) 产出直接写 data/addresses/，下一步跑 compute_features
                   2) 🔴 冲突地址（同时命中 agent 与 bot 判据）从两组都剔除，
                      需人工裁决 —— 这类地址通常是「做自主交易的 agent」，
                      正是三分类最难的部分，值得单独分析
                   3) 人类样本需另行构建，见 data/addresses/README.md

        Returns:
            dict: {'agent': list, 'bot': list}
        """
        log('=' * 60, 'info')
        log('汇总为三组真值地址', 'step')
        log('=' * 60, 'info')

        # ---- agent 正样本 ----
        agent, seen = [], set()
        for fn, tag in self.AGENT_SOURCES:
            rows = read_source_csv(fn)
            n_add = 0
            for r in rows:
                # 🔴 ERC-8004 只收 active，dormant 是空壳
                if tag == 'erc8004' and r.get('tier') != 'active':
                    continue
                a = (r.get('address') or '').lower()
                if self._is_addr(a) and a not in seen:     # 1. 跨源去重
                    seen.add(a)
                    agent.append({'address': a, 'source': r.get('source', tag),
                                  'note': r.get('note', '')})
                    n_add += 1
            log(f'  {fn:<24} 贡献 {n_add:>5} 个（该文件共 {len(rows)} 行）')

        # ---- bot 负样本 ----
        bot, seen_b = [], set()
        for fn, tag in self.BOT_SOURCES:
            rows = read_source_csv(fn)
            for r in rows:
                a = (r.get('address') or '').lower()
                if self._is_addr(a) and a not in seen_b:
                    seen_b.add(a)
                    bot.append({'address': a, 'source': r.get('source', tag),
                                'note': r.get('note', '')})
            log(f'  {fn:<24} 贡献 {len(bot):>5} 个')

        # 2. 🔴 冲突处理
        overlap = seen & seen_b
        if overlap:
            log(f'⚠️ {len(overlap)} 个地址同时命中 agent 与 bot 判据，'
                f'已从两组中剔除（需人工裁决）', 'warn')
            agent = [r for r in agent if r['address'] not in overlap]
            bot = [r for r in bot if r['address'] not in overlap]

        # 3. 写盘
        config.ADDR_DIR.mkdir(parents=True, exist_ok=True)
        for rows, fname, desc in [
            (agent, config.GROUPS['agent'][0],
             'agent 正样本（Olas + ERC8004活跃 + x402）'),
            (bot, config.GROUPS['bot'][0], '传统脚本 bot 负样本（MEV 原子套利）'),
        ]:
            p = config.ADDR_DIR / fname
            with p.open('w', newline='', encoding='utf-8') as f:
                f.write(f'# {desc}\n')
                f.write(f"# 由 FetchOrchestrator 汇总，{time.strftime('%Y-%m-%d %H:%M')}\n")
                f.write('address,source,note\n')
                for r in rows:
                    note = str(r['note']).replace(',', ';')
                    f.write(f"{r['address']},{r['source']},{note}\n")
            log(f'写入 {p.relative_to(config.ROOT)}（{len(rows)} 个）', 'ok')

        log('⚠️ 人类样本需另行构建 —— 见 data/addresses/README.md', 'warn')
        self.merged = {'agent': agent, 'bot': bot, 'overlap': sorted(overlap)}
        return self.merged

    # ============================================================
    #  汇报
    # ============================================================
    def report(self) -> None:
        """打印各数据源的成败与产出行数。"""
        log('=' * 60, 'info')
        log('各数据源结果', 'step')
        for cls, _, _ in self.JOBS:
            if cls.name in self.skip:
                print(f'  ⊘ {cls.name:<14} 跳过')
                continue
            inst = self.instances.get(cls.name)
            outfile = getattr(inst, 'output', cls.output) if inst else cls.output
            n = len(read_source_csv(outfile)) if str(outfile).endswith('.csv') else '-'
            mark = '✅' if self.results.get(cls.name) else '❌'
            print(f'  {mark} {cls.name:<14} {str(n):>6} 行  {cls.desc}')

    def run(self, only_merge: bool = False) -> dict:
        """
        ③ 完整流程：抓取 → 汇报 → 汇总 → 校验

        Args:
            only_merge: True 时只做汇总，不重新抓取
        Returns:
            dict: merge_groups 的结果
        """
        if not only_merge:
            self.fetch_all()
            self.report()
        merged = self.merge_groups()
        self.validate_merged(merged)
        log('下一步：python3 src/feature/compute_features.py', 'ok')
        return merged

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

    def validate_merged(self, merged: dict,
                        context: str = '真值汇总') -> bool:
        """
        校验汇总结果

        检查项:
          1. 两组都非空
          2. 组内无重复、组间无交集
          3. ⚠️ 类别不平衡（影响 macro-F1）
          4. ⚠️ 样本量是否够训练
          5. ⚠️ 冲突地址数量（正是三分类最难的部分）
          6. 🔴 提示: 人类样本缺失

        Args:
            merged:  merge_groups 的结果
            context: 日志上下文名
        Returns:
            bool
        """
        issues, warns = [], []
        agent = merged.get('agent') or []
        bot = merged.get('bot') or []
        overlap = merged.get('overlap') or []

        if not agent:
            issues.append('agent 组为空')
        if not bot:
            issues.append('bot 组为空')
        if issues:
            return self._emit(context, issues, warns)

        a_set = {r['address'] for r in agent}
        b_set = {r['address'] for r in bot}
        if len(a_set) != len(agent):
            issues.append(f'agent 组内有 {len(agent)-len(a_set)} 个重复')
        if len(b_set) != len(bot):
            issues.append(f'bot 组内有 {len(bot)-len(b_set)} 个重复')
        if a_set & b_set:
            issues.append(f'两组仍有 {len(a_set & b_set)} 个交集（冲突剔除失效）')

        # 3. 类别不平衡
        ratio = max(len(agent), len(bot)) / max(min(len(agent), len(bot)), 1)
        if ratio > 5:
            warns.append(f'类别严重不平衡 agent:{len(agent)} vs bot:{len(bot)}'
                         f'（{ratio:.1f}:1）—— 训练需用 class_weight，'
                         f'评估必须看 macro-F1 而非 accuracy')

        for nm, n in (('agent', len(agent)), ('bot', len(bot))):
            if n < 100:
                warns.append(f'{nm} 仅 {n} 个，样本偏少。'
                             f'建议跑 --full 模式扩充')

        if overlap:
            warns.append(f'{len(overlap)} 个冲突地址已剔除 —— 这类「做自主交易的 agent」'
                         f'正是三分类最难的部分，建议单独分析后人工裁决')

        human_file = config.ADDR_DIR / config.GROUPS['human'][0]
        if not human_file.exists() or human_file.stat().st_size < 100:
            warns.append('🔴 人类样本缺失 —— 三分类需要三组，'
                         '见 data/addresses/README.md')
        return self._emit(context, issues, warns)


def main():
    ap = argparse.ArgumentParser(description='一键跑通所有免费数据源并汇总真值')
    ap.add_argument('--full', action='store_true', help='全量模式（慢，约 30 分钟）')
    ap.add_argument('--skip', default='', help='跳过哪些源，逗号分隔')
    ap.add_argument('--only-merge', action='store_true', help='只做汇总，不重新抓取')
    args = ap.parse_args()
    FetchOrchestrator(full=args.full, skip=args.skip).run(only_merge=args.only_merge)


if __name__ == '__main__':
    main()
