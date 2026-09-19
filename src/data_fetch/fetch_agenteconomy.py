#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_agenteconomy.py
@Description:
    agenteconomy.to 聚合数据 —— 趋势交叉验证 + 现成 Dune 查询 ID，附 validate_* 校验。

    ============================================================
    一、 认证与限频
    ============================================================
      Base URL: https://agenteconomy.to/data.json
      认证:     无 —— 公开 JSON，无需 Key、无需注册
      限频:     无
      更新:     每小时刷新（meta.updatedAt 标注实际时间）

      ⚠️ 实测坑: 该站**无视 Accept-Encoding 强推 brotli**，
        urllib 不解压会抛 UnicodeDecodeError: invalid start byte 0x81。
        _base.http_json 已改为显式请求 identity，并保留 brotli 兜底。

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — Agent 经济聚合指标 + 底层 Dune 查询 ID ───────────────────────┐
    │  接口:  GET https://agenteconomy.to/data.json                             │
    │  函数:  fetch_queries() / fetch_series()                                  │
    │  覆盖:  x402 / Base agentic / Virtuals ACP / ERC-8004 / Olas / Tempo MPP  │
    │  更新频率: 每小时                                                          │  ★必填
    │  更新时间: meta.updatedAt 标注（实测约每小时整点后）                       │  ★必填
    │  API KEY: 否 —— 公开 JSON                                                 │  ★必填
    │  历史范围: 各序列自带完整历史（日频 60~245 行 / 周频 52 行 / 月频 12 行）  │  ★必填
    │  产出: data/sources/agenteconomy_dune_queries.csv                         │
    │        data/sources/agenteconomy_raw.json                                 │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 只有聚合口径，**无地址级数据** ⇒ 不能作真值来源，只能做趋势交叉验证。

      ※ 💡 最大价值不是数据本身，而是 meta.queries 公开了底层 Dune 查询 ID：
        这些是现成且经过验证的查询（x402 累计/日度、Base agentic、Virtuals ACP、
        ERC-8004 注册表、Olas、x402 代币分布/链分布），可直接
            https://dune.com/queries/<queryId>
        打开看 SQL 并 fork —— 省掉从零写查询，也能看到官方口径怎么定义。
        附带的 lastCostCredits 还能用来估算自己的 Dune credits 消耗。

      原始字段                    → 统一字段            → 说明
      meta.queries.<name>.queryId → query_id            Dune 查询 ID
      [客户端合成]                 → dune_url            https://dune.com/queries/{id}
      .lastCostCredits            → last_cost_credits   单次执行成本（估算自己消耗用）
      .executedAt                 → executed_at         该查询最近执行时间
      <各序列>[]                   → (raw json)          时间序列，落 raw json 供分析

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. queries 为空 ⇒ 站点结构变更，告警但不报错
      2. 时间序列递归摊平: 只收集 list[dict] 形态的节点，记录 (路径, 行数, 字段)
      3. raw json 原样落盘，不做裁剪（供后续按需分析）

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      📊 **趋势交叉验证**用途：确认自己从 SQD 算出的 x402 笔数/金额曲线
        与业界口径一致，避免自建管线出现系统性偏差。
      💡 **省 Dune credits**：先 fork 这里公开的现成查询，
        免费额度只有 2500 credits/月（导出 20 credits/MB），试错成本很高。
'''

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils._base import (BaseFetcher, http_json, log,  # noqa: E402
                             write_csv, write_json)


# ================================================================
#  AgentEconomyDataSource — agenteconomy.to 聚合数据
# ================================================================
class AgentEconomyDataSource(object):
    """
    agenteconomy.to 数据采集与管理

    核心函数 (共 2 个):
      ① fetch_queries()  提取底层 Dune 查询 ID（本源最大价值）
      ② fetch_series()   摊平可用的时间序列清单

    辅助函数:
      - _load()        拉取并缓存 raw json
      - _walk_series() 递归摊平嵌套 JSON 中的时间序列

    校验函数: validate_queries / validate_raw

    Args:
        无
    """

    URL = 'https://agenteconomy.to/data.json'

    # 预期应该出现的查询名（缺失说明站点结构变更）
    EXPECTED_QUERIES = ['x402Cumulative', 'x402Daily', 'baseAgentic',
                        'erc8004Registry', 'olas']

    def __init__(self):
        """初始化。Returns: None"""
        self.raw = None

    # ============================================================
    #  通用工具
    # ============================================================
    @staticmethod
    def _safe_float(v, default=None):
        """安全转 float: None/空/NaN/不可转 → default。"""
        try:
            if v is None or v == '':
                return default
            fv = float(v)
            return default if fv != fv else fv
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_str(v, default: str = ''):
        """安全转 str: None/NaN → default。"""
        if v is None:
            return default
        s = str(v)
        return default if s.lower() in ('nan', 'none') else s

    def _load(self) -> dict:
        """拉取并缓存 raw json（同一实例内只请求一次）。"""
        if self.raw is None:
            log(f'GET {self.URL}', 'step')
            self.raw = http_json(self.URL, timeout=40)
            log(f"数据更新时间: {self.raw.get('updatedAt')}", 'ok')
        return self.raw

    @classmethod
    def _walk_series(cls, obj, prefix: str = '', out: list = None) -> list:
        """
        递归摊平嵌套 JSON 中的时间序列（清洗逻辑 2）

        Args:
            obj:    待遍历对象
            prefix: 当前路径
            out:    累积结果
        Returns:
            list[(路径, 行数, 字段列表)]
        """
        out = [] if out is None else out
        if isinstance(obj, dict):
            for k, v in obj.items():
                cls._walk_series(v, f'{prefix}.{k}' if prefix else k, out)
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            out.append((prefix, len(obj), list(obj[0].keys())))
        return out

    # ============================================================
    #  ① 底层 Dune 查询 ID
    # ============================================================
    def fetch_queries(self, save_to_csv: bool = True) -> pd.DataFrame:
        """
        ① 提取 agenteconomy.to 公开的底层 Dune 查询 ID

        【数据源】 GET https://agenteconomy.to/data.json → meta.queries
                 - 无需 API Key，公开 JSON
                 - ⚠️ 该站无视 Accept-Encoding 强推 brotli，_base.http_json 已处理

        【指标含义】 一句话: "别人写好并验证过的 Dune 查询，可直接 fork"
                    覆盖 x402 累计/日度、Base agentic、Virtuals ACP、
                    ERC-8004 注册表、Olas 等，是这个站点最大的价值。

        【返回字段】
                  · name              — 查询名（如 x402Cumulative）
                  · query_id          — Dune 查询 ID
                  · dune_url          — https://dune.com/queries/{id}
                  · last_cost_credits — 单次执行成本，用于估算自己的 credits 消耗
                  · executed_at       — 该查询最近执行时间

        【阈值解读】 last_cost_credits 实测 0.2~10.4。
                    Dune 免费层 2500 credits/月 ⇒ 复杂查询跑 240 次就用完。

        【如何使用】
                   1) 打开 dune_url 看 SQL，fork 后改成自己的口径
                   2) 用 src/data_fetch/fetch_dune.py --query-id <id> 直接读缓存结果
                      （不触发执行，几乎不耗 credits）
                   3) 用 last_cost_credits 反推自己的 credits 预算

        Args:
            save_to_csv: 是否写入 data/sources/agenteconomy_dune_queries.csv
        Returns:
            pd.DataFrame
        """
        raw = self._load()
        queries = (raw.get('meta') or {}).get('queries') or {}
        if not queries:
            log('meta.queries 为空 —— 站点结构可能已变更', 'warn')
            return pd.DataFrame()

        log(f'① 底层 Dune 查询（{len(queries)} 个，可直接 fork）:', 'step')
        rows = []
        for nm, q in queries.items():
            qid = q.get('queryId')
            cost = self._safe_float(q.get('lastCostCredits'))
            print(f'  {nm:<20} https://dune.com/queries/{qid}'
                  + (f'   成本 {cost:.2f} credits' if cost else ''))
            rows.append({'name': nm, 'query_id': qid,
                         'dune_url': f'https://dune.com/queries/{qid}',
                         'last_cost_credits': cost,
                         'executed_at': self._safe_str(q.get('executedAt'))})

        df = pd.DataFrame(rows)
        if save_to_csv:
            write_csv('agenteconomy_dune_queries.csv', df.to_dict('records'),
                      list(df.columns),
                      note=('agenteconomy.to 公开的底层 Dune 查询 ID\n'
                            '这些是现成且经过验证的查询，可 fork 后改，省去从零写 SQL\n'
                            'last_cost_credits 可用于估算自己的 Dune credits 消耗'))
        return df

    # ============================================================
    #  ② 可用时间序列清单
    # ============================================================
    def fetch_series(self, save_raw: bool = True, top: int = 15) -> pd.DataFrame:
        """
        ② 摊平 raw json 中的可用时间序列清单

        【数据源】 GET https://agenteconomy.to/data.json → 各嵌套节点

        【指标含义】 一句话: "这个站点到底提供了哪些可用的时间序列"
                    实测 12 组，含 baseAgentic.daily(245 行) / x402.daily(60 行) /
                    olas.weekly(52 行) / erc8004Registry.daily(90 行) 等。

        【返回字段】
                  · path    — JSON 路径（如 x402.daily）
                  · n_rows  — 行数
                  · columns — 字段列表（逗号分隔）

        【如何使用】 📊 趋势交叉验证：把自己从 SQD 算出的 x402 日度笔数曲线
                   与 x402.daily 对比，确认自建管线没有系统性偏差。
                   raw json 已落盘，可按 path 取出具体序列做分析。

        Args:
            save_raw: 是否把 raw json 落盘（data/sources/agenteconomy_raw.json）
            top:      控制台只打印行数最多的前 N 组
        Returns:
            pd.DataFrame
        """
        raw = self._load()
        series = self._walk_series(raw)
        if not series:
            log('未发现任何时间序列', 'warn')
            return pd.DataFrame()

        log(f'② 可用时间序列（{len(series)} 组）:', 'step')
        for path, n, keys in sorted(series, key=lambda x: -x[1])[:top]:
            print(f"  {path:<34} {n:>5} 行  字段: {', '.join(keys[:6])}")

        df = pd.DataFrame([{'path': p, 'n_rows': n, 'columns': ','.join(k)}
                           for p, n, k in sorted(series, key=lambda x: -x[1])])
        if save_raw:
            write_json('agenteconomy_raw.json', raw)
            log('原始 JSON 已存盘，可用于趋势交叉验证', 'ok')
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

    def validate_queries(self, df: pd.DataFrame,
                         context: str = 'Dune 查询 ID') -> bool:
        """
        校验 ① 的产出

        检查项: 非空 / 必需列齐全 / query_id 非空且唯一 /
               ⚠️ 预期查询名是否缺失（站点结构变更的信号）

        Args:
            df / context
        Returns:
            bool
        """
        issues, warns = [], []
        if df is None or df.empty:
            issues.append('结果为空')
            return self._emit(context, issues, warns)

        miss = [c for c in ('name', 'query_id', 'dune_url') if c not in df.columns]
        if miss:
            issues.append(f'缺列: {miss}')
            return self._emit(context, issues, warns)

        if df['query_id'].isna().any():
            issues.append('存在空的 query_id')
        dup = int(df['query_id'].duplicated().sum())
        if dup:
            warns.append(f'{dup} 个 query_id 重复')

        lost = [q for q in self.EXPECTED_QUERIES if q not in set(df['name'])]
        if lost:
            warns.append(f'预期查询缺失: {lost} —— 站点结构可能已变更')

        if 'last_cost_credits' in df.columns:
            tot = df['last_cost_credits'].fillna(0).sum()
            warns.append(f'这些查询全跑一遍约需 {tot:.1f} credits'
                         f'（Dune 免费层 2500/月）')
        return self._emit(context, issues, warns)

    def validate_raw(self, context: str = 'agenteconomy 原始数据') -> bool:
        """
        校验 raw json 的新鲜度与结构

        检查项: 已加载 / 有 updatedAt / ⚠️ 数据是否滞后超过 24 小时
        """
        from datetime import datetime, timezone
        issues, warns = [], []
        if self.raw is None:
            issues.append('尚未加载数据')
            return self._emit(context, issues, warns)

        upd = self._safe_str(self.raw.get('updatedAt'))
        if not upd:
            warns.append('缺少 updatedAt 字段')
        else:
            try:
                t = datetime.fromisoformat(upd.replace('Z', '+00:00'))
                age_h = (datetime.now(timezone.utc) - t).total_seconds() / 3600
                if age_h > 24:
                    warns.append(f'数据已滞后 {age_h:.1f} 小时（该站标称每小时刷新）')
            except Exception:
                warns.append(f'updatedAt 格式无法解析: {upd}')

        n_series = len(self._walk_series(self.raw))
        if n_series < 5:
            warns.append(f'仅摊平出 {n_series} 组序列，明显偏少，站点结构可能已变更')
        return self._emit(context, issues, warns)


# ================================================================
#  AgentEconomyFetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class AgentEconomyFetcher(BaseFetcher):
    """
    薄适配层：把 AgentEconomyDataSource 接进项目的 Fetcher 体系。

    数据逻辑全在 AgentEconomyDataSource；本类只负责参数透传与产出对接。
    """

    name = 'agenteconomy'
    desc = '聚合趋势 + Dune 查询 ID'
    output = 'agenteconomy_dune_queries.csv'
    quality = '—'
    needs_key = False
    fields = ['name', 'query_id', 'dune_url', 'last_cost_credits', 'executed_at']

    def __init__(self, show_queries=False, **kw):
        super().__init__(**kw)
        self.show_queries = bool(show_queries)
        self.ds = AgentEconomyDataSource()
        self.df = pd.DataFrame()

    def fetch(self) -> list:
        self.df = self.ds.fetch_queries(save_to_csv=True)
        self.ds.validate_queries(self.df)
        self.stats['Dune 查询数'] = len(self.df)

        if not self.show_queries:
            s = self.ds.fetch_series(save_raw=True)
            self.ds.validate_raw()
            self.stats['时间序列组数'] = len(s)
        return []      # 已在各方法内落盘

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--show-queries', action='store_true',
                        help='只打印底层 Dune 查询 ID 与链接，不摊平时间序列')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='agenteconomy.to 聚合数据（趋势交叉验证 + Dune 查询 ID）')
    AgentEconomyFetcher.add_args(ap)
    args = ap.parse_args()
    AgentEconomyFetcher(**{k: v for k, v in vars(args).items()
                           if v is not None}).run()


if __name__ == '__main__':
    main()
