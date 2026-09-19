#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_dune.py
@Description:
    Dune Analytics 查询结果拉取 —— 只用于取地址清单，附 validate_* 校验。

    ============================================================
    一、 认证与限频
    ============================================================
      Base URL: https://api.dune.com/api/v1
      认证:     请求头 X-Dune-API-Key
                读取顺序: 构造参数 > 环境变量 DUNE_API_KEY
      注册:     https://dune.com → Settings → API（免费）

      🔴 成本纪律（免费层）:
          每月           2,500 credits
          导出           20 credits/MB  ⇒  免费额度只够导 125 MB
          引擎           Free(2分钟超时) 不计 credits / Medium 10 / Large 20
          超额           $5.00 / 100 credits

        ⚠️ 125 MB 连一天的 Base 全量交易都装不下。
        ⇒ 严格限定用途: ① 取地址清单（几百 KB）② 趋势交叉验证
          主数据必须走 SQD（免费无限制）。

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — Dune 查询结果（读缓存模式）★默认 ────────────────────────────┐
    │  接口:  GET /query/{id}/results?limit={n}                                 │
    │  函数:  fetch_results(query_id, execute=False)                            │
    │  更新频率: 取决于该查询最近一次执行时间                                    │  ★必填
    │  更新时间: 无固定刷新点                                                    │  ★必填
    │  API KEY: 是（免费层即可）                                                │  ★必填
    │  历史范围: 取决于查询本身；本端点只取「最近一次已有结果」                  │  ★必填
    │  产出: data/sources/dune_{query_id}.csv                                   │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 💡 省 credits 的关键: 如果别人（或你自己）刚跑过，直接白拿结果，
        成本远低于 --execute。这是默认模式。

    ┌─ 数据源② — Dune 查询执行（⚠️ 消耗 credits）────────────────────────────┐
    │  接口:  POST /query/{id}/execute → GET /execution/{eid}/status            │
    │         → GET /execution/{eid}/results                                    │
    │  函数:  fetch_results(query_id, execute=True)                             │
    │  API KEY: 是 │ ⚠️ 每次执行都消耗 credits，按引擎档位计费                  │  ★必填
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 轮询等待，默认 5 秒一次、最长 600 秒。

      Dune 原始返回                    → 统一字段  → 说明
      result.rows[]                   → (透传)    查询结果行，列名随查询而定
      result.metadata.total_row_count → (日志)    查询总行数（本次可能只取部分）
      execution_id                    → (日志)    --execute 时的执行 ID

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. 结果为空 ⇒ 该查询可能从未执行过，落 raw json 供排查，提示加 --execute
      2. 列名直接透传（不同查询列结构完全不同，不做统一化）
      3. --execute 时轮询状态，遇 FAILED/CANCELLED 立即抛错

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🔧 **辅助工具**，非核心数据源。
        用途仅限: ① 取地址清单 ② 趋势交叉验证。
      💡 用前先看 data/sources/agenteconomy_dune_queries.csv ——
        那里有 8 个现成且经过验证的查询 ID，fork 它们比从零写省很多 credits。
'''

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils._base import (BaseFetcher, http_json, log,  # noqa: E402
                             write_csv, write_json)


# ================================================================
#  DuneDataSource — Dune Analytics 查询结果
# ================================================================
class DuneDataSource(object):
    """
    Dune Analytics 数据采集与管理（省 credits 模式）

    核心函数 (共 2 个):
      ① fetch_results(query_id, execute=False)  读缓存结果（默认，几乎不耗 credits）
      ② fetch_results(query_id, execute=True)   重新执行（⚠️ 消耗 credits）

    辅助函数:
      - check_quota()        查额度（本调用不耗 credits）
      - _get_latest()        取最近一次已有结果
      - _execute_and_wait()  触发执行并轮询

    校验函数: validate_results

    Args:
        api_key: Dune API Key；缺省读环境变量 DUNE_API_KEY
    """

    API = 'https://api.dune.com/api/v1'

    # 免费层额度常量（用于校验器给出成本提示）
    FREE_CREDITS_PER_MONTH = 2500
    CREDITS_PER_MB_EXPORT = 20
    POLL_INTERVAL = 5
    MAX_WAIT = 600

    def __init__(self, api_key=None):
        """
        初始化

        Args:
            api_key: 构造参数 > 环境变量 DUNE_API_KEY
        Returns:
            None
        """
        self._key = api_key or os.environ.get('DUNE_API_KEY', '').strip()
        self.n_calls = 0
        self.last_meta = {}

    # ============================================================
    #  通用工具
    # ============================================================
    @property
    def key(self) -> str:
        """校验 API Key 存在，缺失时给出可操作的提示。"""
        if not self._key:
            raise SystemExit(
                '\n[配置错误] 缺少 Dune API key。\n'
                '  1) 免费注册 https://dune.com → Settings → API\n'
                '  2) export DUNE_API_KEY=你的key\n'
                '\n  💡 用前先看 data/sources/agenteconomy_dune_queries.csv，\n'
                '     那里有 8 个现成的查询 ID，fork 比从零写省 credits。\n')
        return self._key

    @property
    def headers(self) -> dict:
        return {'X-Dune-API-Key': self.key}

    @staticmethod
    def _safe_int(v, default: int = 0):
        """安全转 int。"""
        try:
            return default if v is None or v == '' else int(v)
        except (TypeError, ValueError):
            return default

    def check_quota(self) -> dict:
        """
        查额度 —— 这个调用本身不消耗 credits。

        Returns:
            dict: 账号信息；端点变更时返回 {}
        """
        try:
            me = http_json(f'{self.API}/auth/me', headers=self.headers)
            self.n_calls += 1
            log('账号信息:', 'ok')
            for k, v in (me or {}).items():
                if isinstance(v, (str, int, float)):
                    print(f'  {k}: {v}')
            return me or {}
        except Exception as e:
            log(f'额度查询失败（端点可能已变更）: {str(e)[:80]}', 'warn')
            log('请直接到 https://dune.com/settings/api 查看剩余 credits', 'info')
            return {}

    def _get_latest(self, query_id: int, limit: int) -> dict:
        """
        取查询的【最近一次已有结果】—— 不触发新执行

        省 credits 的关键: 如果别人（或你自己）刚跑过，直接白拿结果。

        Args:
            query_id / limit
        Returns:
            dict: Dune 原始响应
        """
        log(f'取 query {query_id} 的最新缓存结果（不触发执行）', 'step')
        r = http_json(f'{self.API}/query/{query_id}/results?limit={limit}',
                      headers=self.headers, timeout=120)
        self.n_calls += 1
        return r

    def _execute_and_wait(self, query_id: int, limit: int) -> dict:
        """
        重新执行查询并轮询等待。⚠️ 这会消耗 credits。

        Args:
            query_id / limit
        Returns:
            dict: Dune 原始响应
        Raises:
            RuntimeError / TimeoutError
        """
        log(f'⚠️ 重新执行 query {query_id} —— 这会消耗 credits', 'warn')
        req = urllib.request.Request(
            f'{self.API}/query/{query_id}/execute', data=b'{}',
            headers={**self.headers, 'Content-Type': 'application/json'})
        eid = json.loads(urllib.request.urlopen(req, timeout=60).read())['execution_id']
        self.n_calls += 1
        log(f'  execution_id = {eid}')

        t0 = time.time()
        while time.time() - t0 < self.MAX_WAIT:
            st = http_json(f'{self.API}/execution/{eid}/status', headers=self.headers)
            self.n_calls += 1
            state = st.get('state', '')
            log(f'  状态 {state}（{time.time()-t0:.0f}s）')
            if state == 'QUERY_STATE_COMPLETED':
                r = http_json(f'{self.API}/execution/{eid}/results?limit={limit}',
                              headers=self.headers, timeout=120)
                self.n_calls += 1
                return r
            if state in ('QUERY_STATE_FAILED', 'QUERY_STATE_CANCELLED'):
                raise RuntimeError(f'查询执行失败：{st}')
            time.sleep(self.POLL_INTERVAL)
        raise TimeoutError(f'等待超过 {self.MAX_WAIT}s')

    # ============================================================
    #  ①② Dune 查询结果
    # ============================================================
    def fetch_results(self, query_id: int, execute: bool = False,
                      limit: int = 5000, out_name: str = '',
                      save_to_csv: bool = True) -> pd.DataFrame:
        """
        ①② 拉取 Dune 查询结果

        【数据源】 ① GET  /query/{id}/results        ← 默认，读缓存，几乎不耗 credits
                 ② POST /query/{id}/execute + 轮询 ← --execute，⚠️ 消耗 credits
                 - 需 DUNE_API_KEY（免费层即可）
                 - 免费层 2500 credits/月，导出 20 credits/MB ⇒ 只够导 125 MB

        【指标含义】 一句话: "别人写好的 SQL 的执行结果"
                    Dune 的价值是社区已经把复杂的链上口径封装成了查询，
                    本课题只用它取地址清单和做趋势交叉验证。

        【返回字段】 列名随查询而定，直接透传（不同查询列结构完全不同）

        【阈值解读】 (credits 消耗)
                    · 读缓存      → 几乎为 0
                    · Free 引擎   → 不计 credits（2 分钟超时）
                    · Medium 引擎 → 10 credits/次
                    · Large 引擎  → 20 credits/次
                    · 导出        → 20 credits/MB ← 主要成本来源

        【如何使用】
                   1) 🔴 只用于取地址清单和趋势验证，主数据必须走 SQD
                   2) 默认走读缓存模式；确实需要最新结果才加 --execute
                   3) 先看 data/sources/agenteconomy_dune_queries.csv 里的
                      8 个现成查询，fork 比从零写省很多 credits

        Args:
            query_id:    Dune 查询 ID
            execute:     True 重新执行（消耗 credits）/ False 读缓存
            limit:       最多取多少行
            out_name:    输出文件名；缺省 dune_{query_id}.csv
            save_to_csv: 是否落盘
        Returns:
            pd.DataFrame
        """
        res = (self._execute_and_wait(query_id, limit) if execute
               else self._get_latest(query_id, limit))

        result = res.get('result') or {}
        rows = result.get('rows') or []
        self.last_meta = result.get('metadata') or {}

        if not rows:      # 1. 结果为空
            log('结果为空。该查询可能从未执行过 —— 加 --execute 触发一次', 'warn')
            write_json(f'dune_{query_id}_raw.json', res)
            return pd.DataFrame()

        df = pd.DataFrame(rows)      # 2. 列名直接透传
        total = self._safe_int(self.last_meta.get('total_row_count'))
        log(f'取得 {len(df):,} 行 × {len(df.columns)} 列', 'ok')
        if total:
            log(f'  查询总行数 {total:,}（本次取 {len(df):,}）', 'info')

        if save_to_csv:
            write_csv(out_name or f'dune_{query_id}.csv',
                      df.to_dict('records'), list(df.columns),
                      note=(f'Dune query {query_id} 结果\n'
                            f'https://dune.com/queries/{query_id}\n'
                            f"模式: {'重新执行（消耗 credits）' if execute else '读缓存（不耗 credits）'}"))
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

    def validate_results(self, df: pd.DataFrame,
                         context: str = 'Dune 查询结果') -> bool:
        """
        校验 ①② 的产出

        检查项:
          1. 非空
          2. 全空列（查询可能有误）
          3. ⚠️ 是否被 limit 截断（total_row_count > 实际行数）
          4. ⚠️ 估算导出成本，提示 credits 消耗
          5. 若含 address 列，做地址格式抽查

        Args:
            df / context
        Returns:
            bool
        """
        import re
        issues, warns = [], []

        if df is None or df.empty:
            issues.append('结果为空（该查询可能从未执行过）')
            return self._emit(context, issues, warns)

        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            warns.append(f'{len(empty_cols)} 个全空列: {empty_cols[:5]}')

        total = self._safe_int(self.last_meta.get('total_row_count'))
        if total and total > len(df):
            warns.append(f'结果被截断：查询总行数 {total:,}，本次只取 {len(df):,} 行。'
                         f'需要全量请加大 limit')

        mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        cost = mb * self.CREDITS_PER_MB_EXPORT
        if cost > 1:
            warns.append(f'本次结果约 {mb:.2f} MB，若走导出接口约耗 '
                         f'{cost:.1f} credits（免费层 {self.FREE_CREDITS_PER_MONTH}/月）')

        addr_col = next((c for c in df.columns if 'address' in c.lower()), None)
        if addr_col:
            s = df[addr_col].astype(str).str.lower()
            bad = int((~s.str.match(r'^0x[0-9a-f]{40}$')).sum())
            if bad:
                warns.append(f'{addr_col} 列有 {bad} 个值不是合法 EVM 地址')

        warns.append('🔴 提醒：Dune 只用于取地址清单和趋势验证，主数据必须走 SQD')
        return self._emit(context, issues, warns)


# ================================================================
#  DuneFetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class DuneFetcher(BaseFetcher):
    """
    薄适配层：把 DuneDataSource 接进项目的 Fetcher 体系。

    数据逻辑全在 DuneDataSource；本类只负责参数透传与产出对接。
    """

    name = 'dune'
    desc = 'Dune 查询结果（省 credits 模式）'
    quality = '—'
    needs_key = True

    def __init__(self, query_id=None, execute=False, limit=5000,
                 check=False, out='', **kw):
        super().__init__(**kw)
        self.query_id = int(query_id) if query_id else None
        self.execute = bool(execute)
        self.limit = int(limit or 5000)
        self.check = bool(check)
        self.out = out or ''
        self.ds = DuneDataSource()
        self.df = pd.DataFrame()

    @property
    def output(self):
        return self.out or (f'dune_{self.query_id}.csv' if self.query_id else '')

    def fetch(self) -> list:
        if self.check or not self.query_id:
            self.ds.check_quota()
            if not self.query_id:
                log('未指定 --query-id。可用的现成查询见：', 'info')
                log('  data/sources/agenteconomy_dune_queries.csv', 'info')
                log('  或跑 python3 src/data_fetch/fetch_agenteconomy.py --show-queries',
                    'info')
            return []

        self.df = self.ds.fetch_results(
            query_id=self.query_id, execute=self.execute,
            limit=self.limit, out_name=self.out, save_to_csv=True)
        self.ds.validate_results(self.df)
        self.stats['API 调用次数'] = self.ds.n_calls
        self.stats['结果行数'] = len(self.df)
        return []      # 已在 fetch_results 内落盘

    def run(self, save: bool = True):
        log(f'[{self.name}] {self.desc}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--check', action='store_true', help='只查额度，不消耗 credits')
        ap.add_argument('--query-id', type=int, help='Dune 查询 ID')
        ap.add_argument('--execute', action='store_true',
                        help='重新执行查询（⚠️ 消耗 credits；默认只取缓存结果）')
        ap.add_argument('--limit', type=int, default=5000, help='最多取多少行')
        ap.add_argument('--out', default='', help='输出文件名')
        return ap


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Dune 查询结果拉取（省 credits 模式）')
    DuneFetcher.add_args(ap)
    args = ap.parse_args()
    DuneFetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
