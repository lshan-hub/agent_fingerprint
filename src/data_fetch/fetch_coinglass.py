#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：fetch_coinglass.py
@Description:
    CoinGlass Open API v4 数据采集 —— 9 个指标源统一封装，CSV 落盘，附 validate_* 数据校验。

    ① 交易所链上转账  ② 交易所余额  ③ 现货币种覆盖          ← 课题一直接使用
    ④ 恐慌&贪婪指数  ⑤ AHR999  ⑥ 稳定币总市值
    ⑦ Coinbase 溢价  ⑧ 牛市顶部指标  ⑨ ETF 资金流×元数据    ← 市场环境变量

    ============================================================
    一、 认证与限频
    ============================================================
      Base URL: https://open-api-v4.coinglass.com/api
      认证:     请求头 CG-API-KEY
                读取顺序: 环境变量 COINGLASS_API_KEY > conf/config.COIN_GLASS_KEY
      限频:     Hobbyist 30 req/min / Startup 80 / Standard 300 / Professional 1200
                环境变量 COINGLASS_TIER 指定档位, _throttle 按档位自动节流
      注册:     https://coinglass.com

      ⚠️ 本文件所用端点 Hobbyist 档均可调用，仅 ⑦ 的 1h 粒度需 Standard 起。
      ⚠️ Hobbyist / Startup 是「个人使用」授权。若要开源派生数据集，
        建议只发布经独立链上验证的地址，并注明来源为链上公开数据。

    ============================================================
    二、 数据源与返回字段
    ============================================================

    ┌─ 数据源① — 交易所链上转账 (Exchange On-chain Transfers) ★课题一主力 ─────┐
    │  接口:  GET /exchange/chain/tx/list?symbol={sym}&page={p}&per_page=100    │
    │  函数:  fetch_exchange_chain_tx()                                         │
    │  覆盖:  ⚠️ 仅 Ethereum 链的 ERC-20，且仅「交易所相关」转账                │
    │  更新频率: 实时                                                            │  ★必填
    │  API KEY: 是 (Hobbyist ✅)                                                │  ★必填
    │  历史范围: ⚠️ 滚动窗口，无日期过滤参数，起止由客户端 _filter_by_date 筛   │  ★必填
    │  产出: data/addresses/human_cex.csv (三组真值清单之一)                    │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 只取 transfer_type 含 "out"（交易所 → 用户 = 提币），to_address 即人类样本候选。
      ※ 🔴 CEX 提现只是「人类」的弱先验，不是证明。机构、做市商、甚至 bot 都可能提币。
        必须与 ENS/Farcaster/昼夜节律交叉（至少命中 2 条），并做 5-30% 标签噪声敏感性分析。
      ※ 🔴 只有 Ethereum，而课题主战场是 Base（ETH 12s 出块 ⇒ agent 带 2-10s 物理上不存在）。
        ⇒ 必须做「以太坊提币地址 → Base 活跃度过滤」，副作用是样本偏向跨链活跃用户。

      原始字段          → 统一字段         → 说明
      to_address       → address          提币接收方 (★人类样本候选)
      transfer_type    → (过滤条件)        只取 out*
      amount_usd       → total_usd        按地址累加
      transaction_time → first_ts/last_ts Unix 秒，按地址取 min/max
      exchange_name    → exchanges        去重后的交易所列表
      [按地址计数]      → n_withdrawals    提币笔数

    ┌─ 数据源② — 交易所余额快照 ───────────────────────────────────────────────┐
    │  接口: GET /exchange/balance/list?symbol={sym}    函数: fetch_exchange_balance() │
    │  更新频率: 实时 │ API KEY: 是 (Hobbyist ✅) │ 历史范围: ⚠️ 快照型，无历史 │★必填
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 课题用途: 识别 CEX 热钱包地址，特征计算前从三组样本中剔除
        （交易所热钱包是高频自动化地址，混入会污染 bot 类）。

    ┌─ 数据源③ — 现货支持币种 ─────────────────────────────────────────────────┐
    │  接口: GET /spot/supported-coins                 函数: fetch_supported_coins() │
    │  更新频率: 低频 │ API KEY: 是 (Hobbyist ✅) │ 历史范围: 无（当前快照）    │★必填
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 课题用途: ①最轻量的权限探测 ②确认 PAXG/XAUT 覆盖（课题二 RWA 会用到）。

    ┌─ 数据源④ — 恐慌&贪婪指数 (Fear & Greed Index) ───────────────────────────┐
    │  接口: GET /index/fear-greed-history              函数: fetch_fear_greed() │
    │  更新频率: 日频 T+1（UTC 0 点后更新前一天）                               │★必填
    │  更新时间: 北京时间约 08:00 刷新，建议每日 09:00 后跑增量                  │★必填
    │  API KEY: 是 (Hobbyist ✅) │ 历史范围: 2018-02-01 至今 (~3000 条，单次全量)│★必填
    │  产出: data/sources/coinglass_fear_greed.csv                              │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 返回 dict 含 3 个等长平行数组: time_list(毫秒) / data_list(0~100) / price_list(BTC 价)。
      ※ 分类字段 API 不返回，由客户端按标准阈值 0/25/45/56/75 合成 (_fng_classify)。
      ※ 指标含义: 6 大因子加权 —— 波动率 25% / 市场动量 25% / 社交情绪 15% /
        调查 15% / BTC 占比 10% / 谷歌趋势 10%。巴菲特「别人贪婪我恐惧」的量化版。
      ※ 🎯 课题用途: 作为**外生市场情绪变量**。agent 的链上活跃度可能与市场情绪相关，
        在时序建模时应作为控制变量，避免把「市场热度驱动的活跃」误当成 agent 特征。

    ┌─ 数据源⑤ — AHR999 抄底指标 ──────────────────────────────────────────────┐
    │  接口: GET /index/ahr999                          函数: fetch_ahr999()    │
    │  更新频率: 日频 │ API KEY: 是 (Hobbyist ✅)                               │★必填
    │  历史范围: 2011-02-01 至今 (~5600 条，含 BTC 早期 0.1365 USD 时代)        │★必填
    │  产出: data/sources/coinglass_ahr999.csv                                  │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 返回 list[dict]: date_string("YYYY/MM/DD") / average_price(200日定投均价) /
        ahr999_value / current_value(BTC 现货价)。
      ※ 阈值: <0.45 抄底 / 0.45~1.2 定投 / >1.2 顶部 (_ahr999_classify)。

    ┌─ 数据源⑥ — 稳定币总市值 ─────────────────────────────────────────────────┐
    │  接口: GET /index/stableCoin-marketCap-history  函数: fetch_stablecoin_marketcap() │
    │  更新频率: 日频 │ API KEY: 是 (Hobbyist ✅)                               │★必填
    │  历史范围: 2015-02-25 至今 (~4100 条)                                     │★必填
    │  产出: data/sources/coinglass_stablecoin_mcap.csv                         │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 指标含义: 加密市场的「场内资金池」总量。稳定币增发 = 增量资金进场。
      ※ 🎯 课题用途: x402 用 USDC 结算，稳定币供给是 agent 支付经济的**资金面上界**。

    ┌─ 数据源⑦ — Coinbase 溢价指数 ────────────────────────────────────────────┐
    │  接口: GET /coinbase-premium-index?interval={1d|4h|1h}  函数: fetch_coinbase_premium() │
    │  更新频率: 实时（1d 在 UTC 0 点收盘后定型）                                │★必填
    │  API KEY: 是（1d/4h Hobbyist ✅；⚠️ 1h 需 Standard 起）                  │★必填
    │  历史范围: 单次上限 ~1000 根 K 线 → 1d ≈ 3 年 / 4h ≈ 165 天 / 1h ≈ 42 天  │★必填
    │  产出: data/sources/coinglass_coinbase_premium_{interval}.csv             │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 返回 list[dict]: time(Unix 秒) / premium(USD) / premium_rate(%) / coinbase_price。
      ※ 指标含义: Coinbase(美国机构) 与 Binance(全球零售) 的价差 = 美国资金相对力量。
      ※ 阈值 (premium_rate %): >+0.10 强溢价 / 0~+0.10 温和溢价 /
        -0.10~0 温和折价 / <-0.10 强折价。

    ┌─ 数据源⑧ — 牛市顶部指标 ─────────────────────────────────────────────────┐
    │  接口: GET /bull-market-peak-indicator          函数: fetch_bull_market_peak() │
    │  更新频率: 日频 │ API KEY: 是 (Hobbyist ✅)                               │★必填
    │  历史范围: ⚠️ 无（仅当前快照，历史靠每日抓取累积）                        │★必填
    │  产出: data/sources/coinglass_bull_market_peak.csv                        │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 返回 8-10 个顶部指标 + 各自的触发位与命中状态，用于判断周期位置。

    ┌─ 数据源⑨ — ETF 资金流 × 元数据 (Flow × List JOIN) ───────────────────────┐
    │  接口: GET /etf/{coin}/list         → 全量 ETF 快照（元数据，JOIN 右表）  │
    │        GET /etf/{coin}/flow-history → 资金流分解（时序，JOIN 左表）        │
    │  函数: fetch_etf_flows()                                                  │
    │  覆盖: BTC(~20 只) / ETH(~12 只)；SOL/XRP 端点目前 404（尚无现货 ETF）    │
    │  更新频率: 日频，美股收盘后 1-2 小时（北京时间 08:00~12:00 陆续就绪）      │★必填
    │  API KEY: 是 (Hobbyist ✅) │ 历史范围: ⭐ 全量历史（BTC 2024-01-11 至今） │★必填
    │  产出: data/sources/coinglass_etf_flows_{SYMBOL}.csv                      │
    └───────────────────────────────────────────────────────────────────────────┘
      ※ 两端点客户端 JOIN 后写扁平宽表: JOIN 键 = flow.etf_ticker == list.ticker (UPPER)，
        1 行 = 1 天 × 1 ETF。
      ※ ⚠️ List 是「当前快照」非历史: 静态字段(fund_name/region/list_date)可安全回溯；
        动态字段(aum_usd/nav_usd/premium_discount_percent)在所有历史行均为同一份当前值，
        **仅供参考，勿用于历史回测**。

    ============================================================
    三、 清洗逻辑
    ============================================================
      1. 数据源① 只保留 transfer_type 含 "out" 的记录（提币方向）
      2. to_address 必须是合法 EVM 地址（0x + 40 hex），统一转小写
      3. 按 address 聚合: 笔数 sum / 金额 sum / 交易所去重 / 时间取 min-max
      4. 金额区间过滤 [min_usd, max_usd]: 下限滤粉尘，上限滤机构/做市商
      5. 日频源统一按 date 去重（keep=last）并升序排序

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🔴 CoinGlass 不是必需品。①的用途（CEX 提币 → 人类样本）可用 Dune 的
        labels.cex + 提币接收方替代；④~⑨ 是市场环境变量，属锦上添花。
        课题核心数据（agent 正样本 / bot 负样本）全部来自免费的链上直读 + SQD，
        不依赖本源。
'''

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils._base import BaseFetcher, log, write_csv  # noqa: E402


# ================================================================
#  CoinGlassDataSource — 9 个 CoinGlass 数据源统一封装
# ================================================================
class CoinGlassDataSource(object):
    """
    CoinGlass Open API v4 数据采集与管理

    核心函数 (共 9 个):
      ① fetch_exchange_chain_tx(symbol, pages, ...)   交易所链上转账 → 人类样本候选
      ② fetch_exchange_balance(symbol)                交易所余额快照 → 热钱包识别
      ③ fetch_supported_coins()                       现货币种覆盖 → 权限探测
      ④ fetch_fear_greed(start_time, end_time)        恐慌&贪婪指数 (2018.02 至今)
      ⑤ fetch_ahr999(start_time, end_time)            AHR999 抄底指标 (2011.02 至今)
      ⑥ fetch_stablecoin_marketcap(start, end)        稳定币总市值 (2015.02 至今)
      ⑦ fetch_coinbase_premium(interval, ...)         Coinbase 溢价指数
      ⑧ fetch_bull_market_peak()                      牛市顶部指标 (快照)
      ⑨ fetch_etf_flows(assetList, ...)               ETF 资金流 × 元数据 (全量历史)

    校验函数: validate_chain_tx / validate_exchange_balance / validate_supported_coins
             / validate_daily_series (④⑤⑥⑦ 共用)

    Args:
        cg_api_key: CoinGlass API Key
                    读取顺序: 参数 > 环境变量 COINGLASS_API_KEY > config.COIN_GLASS_KEY
        tier:       档位 hobbyist/startup/standard/professional
                    读取顺序: 参数 > 环境变量 COINGLASS_TIER > config.COIN_GLASS_TIER
    """

    CG_BASE_URL = config.COIN_GLASS_BASE_URL
    TIER_RPM = config.COIN_GLASS_TIER_RPM
    CG_SYMBOL_MAP = config.COIN_GLASS_SYMBOL_MAP

    ADDR_RE = re.compile(r'^0x[0-9a-f]{40}$')

    # 若后续转做课题二 (RWA 预言机偏离检测) 需要确认的标的
    RWA_SYMBOLS = ['PAXG', 'XAUT', 'XAU']

    # 本课题关心的端点 (probe 时逐个测可用性)
    PROBE_ENDPOINTS = [
        ('链上', '/exchange/chain/tx/list', {'symbol': 'USDT', 'per_page': 10},
         '★★ ①: CEX 提币地址 → 人类样本'),
        ('链上', '/exchange/balance/list', {'symbol': 'USDT'}, '②: 交易所热钱包识别'),
        ('现货', '/spot/supported-coins', {}, '③: 权限探测 + PAXG/XAUT 覆盖'),
        ('指数', '/index/fear-greed-history', {}, '④: 恐慌&贪婪（市场情绪控制变量）'),
        ('指数', '/index/ahr999', {}, '⑤: AHR999 抄底指标'),
        ('指数', '/index/stableCoin-marketCap-history', {}, '⑥: 稳定币总市值'),
        ('指数', '/coinbase-premium-index', {'interval': '1d'}, '⑦: Coinbase 溢价'),
        ('指数', '/bull-market-peak-indicator', {}, '⑧: 牛市顶部指标'),
        ('ETF', '/etf/bitcoin/list', {}, '⑨: ETF 元数据'),
    ]

    def __init__(self, cg_api_key=None, tier=None):
        """
        初始化: API Key + 档位限频 + 输出目录

        Args:
            cg_api_key: 参数 > 环境变量 COINGLASS_API_KEY > config.COIN_GLASS_KEY
            tier:       参数 > 环境变量 COINGLASS_TIER > config.COIN_GLASS_TIER
        Returns:
            None
        """
        self.CG_API_KEY = (cg_api_key
                           or os.environ.get('COINGLASS_API_KEY', '').strip()
                           or getattr(config, 'COIN_GLASS_KEY', '').strip())

        self.tier = (tier or os.environ.get('COINGLASS_TIER', '')
                     or config.COIN_GLASS_TIER).strip().lower()
        self.rpm = self.TIER_RPM.get(self.tier, 30)
        self.CG_INTERVAL = 60.0 / self.rpm * 1.2      # 留 20% 余量
        self._last_call = 0.0
        self.n_calls = 0

        self.OUT_DIR = config.DATA_SRC
        self.ADDR_DIR = config.ADDR_DIR

    # ============================================================
    #  通用工具
    # ============================================================
    def _require_key(self):
        """校验 API Key 存在，缺失时给出可操作的提示。"""
        if not self.CG_API_KEY:
            raise SystemExit(
                '\n[配置错误] 缺少 CoinGlass API key。任选其一：\n'
                '  1) export COINGLASS_API_KEY=你的key      （推荐，不进版本库）\n'
                '  2) 填 conf/config.py 的 COIN_GLASS_KEY\n'
                '\n  注册获取: https://coinglass.com\n'
                '\n  ⚠️ 本源非必需 —— 课题核心数据全部来自免费的链上直读 + SQD。\n')

    def _throttle(self):
        """按档位节流。Hobbyist 30/min ⇒ 每 2.4 秒一次，不能更快。"""
        gap = time.monotonic() - self._last_call
        if gap < self.CG_INTERVAL:
            time.sleep(self.CG_INTERVAL - gap)
        self._last_call = time.monotonic()

    def _coinglass_get(self, path, params=None, raise_on_error: bool = True):
        """
        统一 CoinGlass API 调用 (v4)

        Args:
            path:           接口路径（拼在 CG_BASE_URL 后，如 '/index/ahr999'）
            params:         query 参数 dict；缺省 None
            raise_on_error: True 时 HTTP/业务码异常抛 RuntimeError；
                            False 时返回 (ok, payload_or_errmsg, elapsed) 便于批量探测
        Returns:
            raise_on_error=True  → dict: 解析后的 JSON body
            raise_on_error=False → (ok: bool, payload_or_errmsg, elapsed: float)
        """
        self._require_key()
        self._throttle()

        url = f'{self.CG_BASE_URL}{path}'
        if params:
            url += '?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            'accept': 'application/json', 'CG-API-KEY': self.CG_API_KEY})

        t0 = time.time()
        try:
            body = json.loads(urllib.request.urlopen(req, timeout=40).read())
            self.n_calls += 1
        except urllib.error.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode()[:200]
            except Exception:
                pass
            msg = f'HTTP {e.code} {detail}'
            if raise_on_error:
                raise RuntimeError(f'CoinGlass API error: {msg}')
            return False, msg, time.time() - t0
        except Exception as e:
            if raise_on_error:
                raise RuntimeError(f'CoinGlass API error: {e}')
            return False, str(e), time.time() - t0

        # 业务码: 正常为 '0'; 部分端点返回 '200'
        if str(body.get('code')) not in ('0', '200'):
            msg = f"code={body.get('code')}, msg={body.get('msg', '')}"
            if raise_on_error:
                raise RuntimeError(f'CoinGlass API error: {msg}')
            return False, msg, time.time() - t0

        return body if raise_on_error else (True, body.get('data'), time.time() - t0)

    @staticmethod
    def _safe_float(v, default: float = 0.0):
        """安全转 float: None/空/NaN/不可转 → default。"""
        try:
            if v is None or v == '':
                return default
            fv = float(v)
            return default if fv != fv else fv      # NaN
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_str(v, default: str = ''):
        """安全转 str: None/NaN → default。"""
        if v is None:
            return default
        s = str(v)
        return default if s.lower() in ('nan', 'none') else s

    @staticmethod
    def _parse_date(d):
        """把 'YYYY-MM-DD' / datetime / Unix 秒 统一转成 Unix 秒 (UTC)。"""
        if d is None:
            return None
        if isinstance(d, (int, float)):
            return int(d)
        if isinstance(d, datetime):
            return int(d.replace(tzinfo=timezone.utc).timestamp())
        return int(datetime.strptime(str(d)[:10], '%Y-%m-%d')
                   .replace(tzinfo=timezone.utc).timestamp())

    def _filter_by_date(self, df: pd.DataFrame, start_time, end_time,
                        col: str = 'last_ts') -> pd.DataFrame:
        """按时间窗筛选（本 API 多数端点无日期参数，只能客户端筛）。"""
        if df.empty or col not in df.columns:
            return df
        st, et = self._parse_date(start_time), self._parse_date(end_time)
        if st is not None:
            df = df[df[col] >= st]
        if et is not None:
            df = df[df[col] <= et + 86399]      # 含当日
        return df.reset_index(drop=True)

    @staticmethod
    def _filter_by_date_str(df: pd.DataFrame, start_time, end_time,
                            col: str = 'date') -> pd.DataFrame:
        """按 YYYY-MM-DD 字符串列筛选（日频源用）。"""
        if df.empty or col not in df.columns:
            return df
        if start_time:
            df = df[df[col] >= str(start_time)[:10]]
        if end_time:
            df = df[df[col] <= str(end_time)[:10]]
        return df.reset_index(drop=True)

    @staticmethod
    def _now_str():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _save_csv(self, df: pd.DataFrame, filename: str, note: str = '') -> Path:
        """落盘到 data/sources/。"""
        return write_csv(filename, df.to_dict('records'), list(df.columns), note=note)

    # ============================================================
    #  探测: 本档位实际能访问哪些端点
    # ============================================================
    def probe(self) -> pd.DataFrame:
        """
        探测本档位实际能访问哪些端点 —— 不要信定价页，直接测。

        Returns:
            pd.DataFrame: columns = [category, path, why, ok, detail, elapsed_s]
        """
        log(f'探测端点可用性（档位 {self.tier} = {self.rpm} req/min，'
            f'{len(self.PROBE_ENDPOINTS)} 个端点约 '
            f'{len(self.PROBE_ENDPOINTS)*self.CG_INTERVAL:.0f}s）', 'step')
        rows = []
        for cat, path, params, why in self.PROBE_ENDPOINTS:
            ok, res, el = self._coinglass_get(path, params, raise_on_error=False)
            n = len(res) if ok and isinstance(res, (list, dict)) else None
            print(f"  {'✅' if ok else '❌'} [{cat}] {path}")
            print(f'       {why}')
            print(f'       {el:.2f}s' + (f' | {n} 项' if n is not None else '')
                  if ok else f'       ⚠️  {res}')
            rows.append({'category': cat, 'path': path, 'why': why, 'ok': bool(ok),
                         'detail': '' if ok else str(res), 'elapsed_s': round(el, 3)})
        df = pd.DataFrame(rows)
        log(f"可用 {int(df['ok'].sum())}/{len(df)} 个端点", 'ok')
        return df

    # ============================================================
    #  ① 交易所链上转账 → 人类样本候选
    # ============================================================
    def _fetch_chain_tx_api(self, symbol: str, pages: int) -> list:
        """
        拉取交易所链上转账原始记录（翻页）

        API: GET /exchange/chain/tx/list?symbol={sym}&page={p}&per_page=100

        Args:
            symbol: ERC-20 代币符号（如 'USDT' / 'USDC'）
            pages:  最多翻多少页（每页 100 条）
        Returns:
            list[dict]: 原始记录；遇空页即停止
        """
        out = []
        for page in range(1, pages + 1):
            body = self._coinglass_get('/exchange/chain/tx/list', {
                'symbol': symbol, 'page': page, 'per_page': 100})
            data = body.get('data')
            rows = (data if isinstance(data, list)
                    else data.get('list', []) if isinstance(data, dict) else [])
            if not rows:
                break
            out.extend(rows)
            if page % 5 == 0 or page == pages:
                log(f'  第 {page}/{pages} 页 | 累计 {len(out)} 条原始记录')
        return out

    def _aggregate_withdrawals(self, raw: list) -> pd.DataFrame:
        """
        按接收地址聚合提币记录（清洗逻辑见模块文档 三）

        Returns:
            pd.DataFrame: [address, n_withdrawals, total_usd, exchanges, first_ts, last_ts]
        """
        agg, n_out = {}, 0
        for r in raw:
            if 'out' not in self._safe_str(r.get('transfer_type')).lower():
                continue                                   # 1. 只保留提币方向
            to = self._safe_str(r.get('to_address')).lower()
            if not self.ADDR_RE.match(to):
                continue                                   # 2. 地址合法性
            n_out += 1
            s = agg.setdefault(to, {'address': to, 'n_withdrawals': 0,
                                    'total_usd': 0.0, '_ex': set(),
                                    'first_ts': None, 'last_ts': None})
            s['n_withdrawals'] += 1                        # 3. 按地址聚合
            s['total_usd'] += self._safe_float(r.get('amount_usd'))
            ex = self._safe_str(r.get('exchange_name'))
            if ex:
                s['_ex'].add(ex)
            ts = int(self._safe_float(r.get('transaction_time')))
            if ts:
                s['first_ts'] = ts if s['first_ts'] is None else min(s['first_ts'], ts)
                s['last_ts'] = ts if s['last_ts'] is None else max(s['last_ts'], ts)

        log(f'  原始 {len(raw)} 条 → 提币 {n_out} 条 → 去重地址 {len(agg)} 个', 'info')
        if not agg:
            return pd.DataFrame()
        for s in agg.values():
            s['exchanges'] = '|'.join(sorted(s.pop('_ex'))[:3]) or 'NA'
        return pd.DataFrame(list(agg.values()))[
            ['address', 'n_withdrawals', 'total_usd', 'exchanges', 'first_ts', 'last_ts']]

    def fetch_exchange_chain_tx(self, symbol: str = 'USDT', pages: int = 20,
                                start_time=None, end_time=None,
                                min_usd: float = 100.0, max_usd: float = 2_000_000.0,
                                save_to_csv: bool = True,
                                append: bool = False) -> pd.DataFrame:
        """
        ① 获取交易所链上转账并提取提币接收方（人类样本候选）

        【数据源】 GET /exchange/chain/tx/list?symbol={sym}&page={p}&per_page=100
                 - 需 CG_API_KEY (Hobbyist ✅)
                 - 覆盖: ⚠️ 仅 Ethereum 链的 ERC-20，且仅「交易所相关」转账
                 - 历史范围: ⚠️ 滚动窗口，无日期过滤参数，起止由客户端筛
                 返回 list[dict]，每条形如:
                    {"from_address": "0x...",        ← 交易所热钱包
                     "to_address":   "0x...",        ← ★提币接收方（人类样本候选）
                     "transfer_type": "outflow",
                     "amount_usd": 1234.56,
                     "exchange_name": "Binance",
                     "transaction_time": 1788400000} ← Unix 秒

        【指标含义】 一句话: "从交易所提币的地址，大概率是自然人"
                    交易所提币必然经过 KYC ⇒ 对「这个地址背后是人」提供弱先验。

        【返回字段】 address / n_withdrawals / total_usd / exchanges / first_ts / last_ts

        【阈值解读】 (total_usd)
                    · < $100     → 粉尘/空投，剔除
                    · $100~$2M   → 自然人合理区间（默认收录）
                    · > $2M      → 机构/做市商，剔除

        【如何使用】
                   1) 🔴 必须交叉验证: 还需命中 ENS/Farcaster 或昼夜节律，至少 2 条信号
                   2) 🔴 必须跨链关联: 本源只有 Ethereum，主战场是 Base
                   3) 🔴 必须做敏感性分析: 注入 5-30% 标签噪声验证结论稳定性
                   4) 人工抽验 30 个，估计真实纯度，写进论文的标签质量一节

        Args:
            symbol / pages / start_time / end_time / min_usd / max_usd
            save_to_csv: 是否写入 data/addresses/human_cex.csv
            append:      True 追加 / False 覆盖
        Returns:
            pd.DataFrame
        """
        log(f'① 拉取 {symbol} 的交易所链上转账（{pages} 页）', 'step')
        log(f'档位 {self.tier} = {self.rpm} req/min，'
            f'预计 {pages * self.CG_INTERVAL / 60:.1f} 分钟', 'info')

        raw = self._fetch_chain_tx_api(symbol, pages)
        if not raw:
            log('没有拿到任何转账记录。先跑 probe() 确认端点权限。', 'err')
            return pd.DataFrame()

        df = self._aggregate_withdrawals(raw)
        if df.empty:
            log('没有提币方向的记录', 'warn')
            return df

        df = self._filter_by_date(df, start_time, end_time)
        before = len(df)
        df = df[(df['total_usd'] >= min_usd) & (df['total_usd'] <= max_usd)]
        df = df.sort_values('address').reset_index(drop=True)
        log(f'  金额过滤 [{min_usd:,.0f}, {max_usd:,.0f}] USD: '
            f'{before} → {len(df)} 个地址', 'ok')

        if save_to_csv and not df.empty:
            self._save_human_sample(df, symbol, append)
        return df

    def _save_human_sample(self, df: pd.DataFrame, symbol: str,
                           append: bool = False) -> Path:
        """写入三组真值清单之一: data/addresses/human_cex.csv"""
        out = self.ADDR_DIR / config.GROUPS['human'][0]
        mode = 'a' if (append and out.exists() and out.stat().st_size) else 'w'
        with out.open(mode, newline='', encoding='utf-8') as f:
            if mode == 'w':
                f.write('# 人类样本候选 —— 由 CoinGlassDataSource 生成\n')
                f.write(f'# 时间: {self._now_str()}\n')
                f.write('# 🔴 CEX 提现只是弱先验，不是证明。必须与 ENS/Farcaster/昼夜节律\n')
                f.write('#    交叉验证，并做 5-30% 标签噪声敏感性分析。\n')
                f.write('# 🔴 本源只有 Ethereum，而主战场是 Base —— 需做跨链活跃度过滤。\n')
                f.write('address,source,note\n')
            for r in df.to_dict('records'):
                note = (f"n={r['n_withdrawals']} usd={r['total_usd']:.0f} "
                        f"ex={r['exchanges']}").replace(',', ';')
                f.write(f"{r['address']},coinglass_cex_withdrawal_{symbol},{note}\n")

        log(f"写入 {out.relative_to(config.ROOT)}"
            f"（{'追加' if mode == 'a' else '覆盖'}）：{len(df)} 个地址", 'ok')
        print()
        log('下一步（必做，不要跳过）：', 'warn')
        print('  1. 交叉验证：还需命中 ENS/Farcaster 或昼夜节律，至少 2 条信号')
        print('  2. ⚠️ 本源只有 Ethereum ERC-20，而主战场是 Base ——')
        print('     需确认这些地址在 Base 上是否也活跃，否则拉不到足够交易')
        print('  3. 人工抽验 30 个，估计这一组的真实纯度，写进论文的标签质量一节')
        return out

    # ============================================================
    #  ② 交易所余额快照
    # ============================================================
    def fetch_exchange_balance(self, symbol: str = 'USDT',
                               save_to_csv: bool = True) -> pd.DataFrame:
        """
        ② 获取各交易所持有的指定代币余额快照

        【数据源】 GET /exchange/balance/list?symbol={sym}
                 - 需 CG_API_KEY (Hobbyist ✅) │ 更新频率: 实时 │ 历史范围: ⚠️ 快照型

        【指标含义】 一句话: "各交易所的链上储备规模"
                    本课题不关心余额数值本身，关心它附带的交易所标识 ——
                    用于把 CEX 热钱包从行为样本中剔除（热钱包是高频自动化地址，
                    混入会污染 bot 类）。

        【如何使用】 与 ① 的 from_address 交叉，建立交易所热钱包清单，
                   在特征计算前从三组样本中排除。

        Args:
            symbol / save_to_csv
        Returns:
            pd.DataFrame
        """
        log(f'② 拉取 {symbol} 的交易所余额快照', 'step')
        body = self._coinglass_get('/exchange/balance/list', {'symbol': symbol})
        data = body.get('data')
        rows = (data if isinstance(data, list)
                else data.get('list', []) if isinstance(data, dict) else [])
        if not rows:
            log('返回为空', 'warn')
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['symbol'] = symbol
        df['update_time'] = self._now_str()
        log(f'  {len(df)} 家交易所 × {len(df.columns)} 列', 'ok')

        if save_to_csv:
            self._save_csv(df, 'coinglass_exchange_balance.csv',
                           note=(f'CoinGlass 交易所 {symbol} 余额快照\n'
                                 '课题用途：识别 CEX 热钱包，特征计算前从样本中剔除'))
        return df

    # ============================================================
    #  ③ 现货支持币种
    # ============================================================
    def fetch_supported_coins(self, save_to_csv: bool = True) -> pd.DataFrame:
        """
        ③ 获取现货支持币种列表

        【数据源】 GET /spot/supported-coins
                 - 需 CG_API_KEY (Hobbyist ✅) │ 历史范围: 无（当前快照）

        【指标含义】 一句话: "本档位 API 是否正常 + RWA 标的是否覆盖"
                    ①最轻量的权限探测端点
                    ②确认 PAXG/XAUT 等黄金 RWA 标的覆盖（课题二会用到）

        【返回字段】 symbol / is_rwa_target（客户端合成）

        Args:
            save_to_csv
        Returns:
            pd.DataFrame
        """
        log('③ 拉取现货支持币种', 'step')
        body = self._coinglass_get('/spot/supported-coins')
        coins = body.get('data') or []
        if not isinstance(coins, list) or not coins:
            log('返回为空', 'warn')
            return pd.DataFrame()

        upper = [self._safe_str(c).upper() for c in coins]
        df = pd.DataFrame({'symbol': upper})
        df['is_rwa_target'] = df['symbol'].isin(self.RWA_SYMBOLS)

        log(f'  支持 {len(df)} 个币种', 'ok')
        for s in self.RWA_SYMBOLS:
            print(f"    {'✅' if s in upper else '❌'} {s}")

        if save_to_csv:
            self._save_csv(df, 'coinglass_supported_coins.csv',
                           note=('CoinGlass 现货支持币种\n'
                                 'is_rwa_target 标记课题二关注的黄金 RWA 标的'))
        return df

    # ============================================================
    #  ④ 恐慌&贪婪指数
    # ============================================================
    @staticmethod
    def _fng_classify(v):
        """FGI 数值 → 分类（客户端按标准阈值 0/25/45/56/75 合成）。"""
        v = int(round(v))
        if v <= 24:
            return 'Extreme Fear'
        if v <= 44:
            return 'Fear'
        if v <= 55:
            return 'Neutral'
        if v <= 74:
            return 'Greed'
        return 'Extreme Greed'

    def fetch_fear_greed(self, start_time=None, end_time=None,
                         save_to_csv: bool = True) -> pd.DataFrame:
        """
        ④ 获取恐慌&贪婪指数全量历史

        【数据源】 GET /index/fear-greed-history
                 - 需 CG_API_KEY (Hobbyist ✅)
                 - 更新频率: 日频 T+1（UTC 0 点后更新前一天；北京时间约 08:00 刷新）
                 - 历史范围: 2018-02-01 至今 (~3000 条)，单次全量返回无需分页
                 返回 dict 含 3 个等长平行数组:
                   time_list(毫秒) / data_list(FGI 0~100) / price_list(当日 BTC 价)

        【指标含义】 一句话: "短期市场情绪温度计 + 反向操作工具"
                    6 大因子加权 —— 波动率 25% / 市场动量 25% / 社交情绪 15% /
                    调查 15% / BTC 占比 10% / 谷歌趋势 10%

        【返回字段】 date / fng_value(int) / fng_classification / btc_price /
                    unix_timestamp(秒) / update_time

        【阈值解读】 0~24 Extreme Fear（买入机会）/ 25~44 Fear / 45~55 Neutral /
                    56~74 Greed / 75~100 Extreme Greed（卖出机会）

        【如何使用】 🎯 课题用途: 作为**外生市场情绪控制变量**。
                   agent 的链上活跃度可能随市场情绪波动，时序建模时应控制此变量，
                   避免把「市场热度驱动的活跃」误当成 agent 的行为特征。

        Args:
            start_time / end_time: 'YYYY-MM-DD'；None 不限
            save_to_csv
        Returns:
            pd.DataFrame
        """
        log('④ 拉取恐慌&贪婪指数', 'step')
        d = self._coinglass_get('/index/fear-greed-history').get('data') or {}
        times, values = d.get('time_list') or [], d.get('data_list') or []
        prices = d.get('price_list') or []
        if not times or not values:
            log('返回为空', 'warn')
            return pd.DataFrame()

        # 平行数组长度不等时截齐，避免 pd.DataFrame 抛 ValueError
        n = min(len(times), len(values))
        times, values = times[:n], values[:n]
        prices = (prices[:n] + [None] * n)[:n]

        df = pd.DataFrame({
            'ts_ms': [int(t) for t in times],
            'v': [self._safe_float(v, None) for v in values],
            'btc_price': [self._safe_float(p, None) for p in prices],
        }).dropna(subset=['v'])

        df['date'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.strftime('%Y-%m-%d')
        df['fng_value'] = df['v'].round().astype(int)
        df['fng_classification'] = df['fng_value'].apply(self._fng_classify)
        df['unix_timestamp'] = (df['ts_ms'] // 1000).astype('int64')
        df['update_time'] = self._now_str()

        df = (df[['date', 'fng_value', 'fng_classification', 'btc_price',
                  'unix_timestamp', 'update_time']]
              .drop_duplicates(subset=['date'], keep='last')
              .sort_values('date').reset_index(drop=True))
        df = self._filter_by_date_str(df, start_time, end_time)
        log(f'  {len(df)} 条（{df["date"].min()} ~ {df["date"].max()}）', 'ok')

        if save_to_csv:
            self._save_csv(df, 'coinglass_fear_greed.csv',
                           note=('CoinGlass 恐慌&贪婪指数（日频，2018-02 至今）\n'
                                 '课题用途：外生市场情绪控制变量'))
        return df

    # ============================================================
    #  ⑤ AHR999 抄底指标
    # ============================================================
    @staticmethod
    def _ahr999_classify(v):
        """AHR999 → 三档操作建议: 抄底(<0.45) / 定投(0.45~1.2) / 顶部(>1.2)。"""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ''
        if v < 0.45:
            return '抄底'
        if v <= 1.2:
            return '定投'
        return '顶部'

    def fetch_ahr999(self, start_time=None, end_time=None,
                     save_to_csv: bool = True) -> pd.DataFrame:
        """
        ⑤ 获取 AHR999 抄底指标全量历史

        【数据源】 GET /index/ahr999
                 - 需 CG_API_KEY (Hobbyist ✅) │ 更新频率: 日频
                 - 历史范围: 2011-02-01 至今 (~5600 条，含 BTC 早期 0.1365 USD 时代)
                 返回 list[dict]: date_string("YYYY/MM/DD") / average_price(200日定投均价)
                                 / ahr999_value / current_value(BTC 现货价)

        【指标含义】 一句话: "BTC 定投择时指标" = (价格/200日定投成本) × (价格/指数增长估值)

        【返回字段】 date / ahr999_value / ma200_price / btc_price / category /
                    unix_timestamp / update_time

        【阈值解读】 <0.45 抄底区 / 0.45~1.2 定投区 / >1.2 顶部区

        Args:
            start_time / end_time / save_to_csv
        Returns:
            pd.DataFrame
        """
        log('⑤ 拉取 AHR999 指标', 'step')
        data = self._coinglass_get('/index/ahr999').get('data') or []
        if not data:
            log('返回为空', 'warn')
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date_string'],
                                    format='%Y/%m/%d').dt.strftime('%Y-%m-%d')
        df['ahr999_value'] = pd.to_numeric(df['ahr999_value'], errors='coerce')
        df['ma200_price'] = pd.to_numeric(df['average_price'], errors='coerce')
        df['btc_price'] = pd.to_numeric(df['current_value'], errors='coerce')
        df['category'] = df['ahr999_value'].apply(self._ahr999_classify)
        df['unix_timestamp'] = (pd.to_datetime(df['date']).astype('int64') // 10 ** 9)
        df['update_time'] = self._now_str()

        df = (df[['date', 'ahr999_value', 'ma200_price', 'btc_price',
                  'category', 'unix_timestamp', 'update_time']]
              .dropna(subset=['ahr999_value'])
              .drop_duplicates(subset=['date'], keep='last')
              .sort_values('date').reset_index(drop=True))
        df = self._filter_by_date_str(df, start_time, end_time)
        log(f'  {len(df)} 条（{df["date"].min()} ~ {df["date"].max()}）', 'ok')

        if save_to_csv:
            self._save_csv(df, 'coinglass_ahr999.csv',
                           note='CoinGlass AHR999 抄底指标（日频，2011-02 至今）')
        return df

    # ============================================================
    #  ⑥ 稳定币总市值
    # ============================================================
    def fetch_stablecoin_marketcap(self, start_time=None, end_time=None,
                                   save_to_csv: bool = True) -> pd.DataFrame:
        """
        ⑥ 获取稳定币总市值全量历史

        【数据源】 GET /index/stableCoin-marketCap-history
                 - 需 CG_API_KEY (Hobbyist ✅) │ 更新频率: 日频
                 - 历史范围: 2015-02-25 至今 (~4100 条，含 USDT 早期仅 30 万 USD 的草创期)
                 返回 dict 含平行数组 time_list(毫秒) + data_list / market_cap_list

        【指标含义】 一句话: "加密市场的场内资金池总量"
                    稳定币增发 = 增量资金进场；缩量 = 资金撤离。

        【返回字段】 date / stablecoin_mcap_usd / unix_timestamp / update_time

        【如何使用】 🎯 课题用途: x402 用 USDC 结算，稳定币供给是 agent 支付经济的
                   **资金面上界**。可作为 agent 经济规模外推时的合理性上界校验。

        Args:
            start_time / end_time / save_to_csv
        Returns:
            pd.DataFrame
        """
        log('⑥ 拉取稳定币总市值', 'step')
        d = self._coinglass_get('/index/stableCoin-marketCap-history').get('data') or {}

        # 该端点返回结构随版本变化，兼容 dict-of-arrays 与 list-of-dicts 两种
        if isinstance(d, dict):
            times = d.get('time_list') or []
            caps = (d.get('data_list') or d.get('market_cap_list')
                    or d.get('marketcap_list') or [])
            n = min(len(times), len(caps))
            rows = [{'ts_ms': int(times[i]), 'cap': self._safe_float(caps[i], None)}
                    for i in range(n)]
        else:
            rows = [{'ts_ms': int(self._safe_float(r.get('time') or r.get('timestamp'))),
                     'cap': self._safe_float(r.get('market_cap')
                                             or r.get('marketCap'), None)}
                    for r in (d or [])]
        rows = [r for r in rows if r['cap'] is not None and r['ts_ms']]
        if not rows:
            log('返回为空或结构不识别', 'warn')
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # 毫秒/秒自适应
        if df['ts_ms'].max() < 1e11:
            df['ts_ms'] = df['ts_ms'] * 1000
        df['date'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.strftime('%Y-%m-%d')
        df['stablecoin_mcap_usd'] = df['cap']
        df['unix_timestamp'] = (df['ts_ms'] // 1000).astype('int64')
        df['update_time'] = self._now_str()

        df = (df[['date', 'stablecoin_mcap_usd', 'unix_timestamp', 'update_time']]
              .drop_duplicates(subset=['date'], keep='last')
              .sort_values('date').reset_index(drop=True))
        df = self._filter_by_date_str(df, start_time, end_time)
        log(f'  {len(df)} 条（{df["date"].min()} ~ {df["date"].max()}）', 'ok')

        if save_to_csv:
            self._save_csv(df, 'coinglass_stablecoin_mcap.csv',
                           note=('CoinGlass 稳定币总市值（日频，2015-02 至今）\n'
                                 '课题用途：agent 支付经济的资金面上界'))
        return df

    # ============================================================
    #  ⑦ Coinbase 溢价指数
    # ============================================================
    @staticmethod
    def _coinbase_premium_classify(rate):
        """premium_rate(%) → 强溢价 / 溢价 / 折价 / 强折价。"""
        try:
            r = float(rate)
        except (TypeError, ValueError):
            return ''
        if r > 0.10:
            return '强溢价'
        if r > 0:
            return '溢价'
        if r > -0.10:
            return '折价'
        return '强折价'

    def fetch_coinbase_premium(self, interval: str = '1d',
                               start_time=None, end_time=None,
                               save_to_csv: bool = True) -> pd.DataFrame:
        """
        ⑦ 获取 Coinbase 溢价指数

        【数据源】 GET /coinbase-premium-index?interval={1d|4h|1h}
                 - 1d / 4h : Hobbyist ✅
                 - 1h      : ⚠️ 需 Standard 起
                 - 历史范围: 单次上限 ~1000 根 K 线
                            1d ≈ 3 年 / 4h ≈ 165 天 / 1h ≈ 42 天
                 返回 list[dict]:
                    {"time": 1780358400,        ← Unix 秒（注意是秒不是毫秒）
                     "premium": -94.47,         ← Coinbase价 − Binance价 (USD)
                     "premium_rate": -0.1323,   ← premium / Binance价 (%，已乘100)
                     "coinbase_price": 71314.43}

        【指标含义】 一句话: "美国机构资金 vs 全球零售资金的力量对比表"
                    · Coinbase: 美国合规交易所（机构 + 美国散户）
                    · Binance:  全球零售为主
                    · 正溢价 → 美国买盘强势 / 负溢价 → 美国卖盘强势

        【返回字段】 date / interval_type / premium / premium_rate /
                    coinbase_price / category / unix_timestamp / update_time

        【阈值解读】 >+0.10% 强溢价 / 0~+0.10% 溢价 / -0.10%~0 折价 / <-0.10% 强折价

        Args:
            interval: '1d' / '4h' / '1h'（1h 需 Standard）
            start_time / end_time / save_to_csv
        Returns:
            pd.DataFrame
        """
        if interval == '1h' and self.tier in ('hobbyist', 'startup'):
            log(f'⚠️ interval=1h 需 Standard 起，当前档位 {self.tier} 可能返回 403', 'warn')

        log(f'⑦ 拉取 Coinbase 溢价指数（interval={interval}）', 'step')
        data = self._coinglass_get('/coinbase-premium-index',
                                   {'interval': interval}).get('data') or []
        if not data:
            log('返回为空', 'warn')
            return pd.DataFrame()

        df = pd.DataFrame(data)
        ts = pd.to_numeric(df.get('time'), errors='coerce')
        fmt = '%Y-%m-%d' if interval == '1d' else '%Y-%m-%d %H:%M:%S'
        df['date'] = pd.to_datetime(ts, unit='s').dt.strftime(fmt)
        df['interval_type'] = interval
        df['premium'] = pd.to_numeric(df.get('premium'), errors='coerce')
        df['premium_rate'] = pd.to_numeric(df.get('premium_rate'), errors='coerce')
        df['coinbase_price'] = pd.to_numeric(df.get('coinbase_price'), errors='coerce')
        df['category'] = df['premium_rate'].apply(self._coinbase_premium_classify)
        df['unix_timestamp'] = ts.astype('int64')
        df['update_time'] = self._now_str()

        df = (df[['date', 'interval_type', 'premium', 'premium_rate',
                  'coinbase_price', 'category', 'unix_timestamp', 'update_time']]
              .dropna(subset=['premium_rate'])
              .drop_duplicates(subset=['date'], keep='last')
              .sort_values('date').reset_index(drop=True))
        df = self._filter_by_date_str(df, start_time, end_time)
        log(f'  {len(df)} 根 K 线（{df["date"].min()} ~ {df["date"].max()}）', 'ok')

        if save_to_csv:
            self._save_csv(df, f'coinglass_coinbase_premium_{interval}.csv',
                           note=f'CoinGlass Coinbase 溢价指数（{interval}）')
        return df

    # ============================================================
    #  ⑧ 牛市顶部指标
    # ============================================================
    def fetch_bull_market_peak(self, save_to_csv: bool = True) -> pd.DataFrame:
        """
        ⑧ 获取牛市顶部综合指标（快照）

        【数据源】 GET /bull-market-peak-indicator
                 - 需 CG_API_KEY (Hobbyist ✅) │ 更新频率: 日频
                 - 历史范围: ⚠️ 无（仅当前快照，历史靠每日抓取累积）
                 返回 list[dict]，8-10 个指标，每条含指标名 / 当前值 / 触发位 / 命中状态

        【指标含义】 一句话: "多个顶部信号的综合看板"
                    每个指标有各自的触发阈值，命中数量越多越接近周期顶部。

        【返回字段】 随 API 结构透传 + snapshot_date / update_time

        Args:
            save_to_csv
        Returns:
            pd.DataFrame
        """
        log('⑧ 拉取牛市顶部指标', 'step')
        data = self._coinglass_get('/bull-market-peak-indicator').get('data') or []
        if not data:
            log('返回为空', 'warn')
            return pd.DataFrame()

        df = pd.DataFrame(data if isinstance(data, list) else [data])
        df['snapshot_date'] = datetime.now().strftime('%Y-%m-%d')
        df['update_time'] = self._now_str()

        hit_col = next((c for c in df.columns
                        if 'hit' in c.lower() or 'trigger' in c.lower()), None)
        if hit_col is not None:
            try:
                n_hit = int(df[hit_col].astype(str).str.lower()
                            .isin(['true', '1', 'yes']).sum())
                log(f'  {len(df)} 个指标，其中 {n_hit} 个已触发', 'ok')
            except Exception:
                log(f'  {len(df)} 个指标', 'ok')
        else:
            log(f'  {len(df)} 个指标 × {len(df.columns)} 列', 'ok')

        if save_to_csv:
            self._save_csv(df, 'coinglass_bull_market_peak.csv',
                           note=('CoinGlass 牛市顶部指标（快照型）\n'
                                 '⚠️ 无历史，需每日抓取累积'))
        return df

    # ============================================================
    #  ⑨ ETF 资金流 × 元数据
    # ============================================================
    def _fetch_etf_list(self, symbol: str) -> dict:
        """
        获取 ETF 全量快照（JOIN 右表）

        API: GET /etf/{coin_name}/list

        Returns:
            dict: {TICKER_UPPER: {...全字段...}}
        """
        coin = self.CG_SYMBOL_MAP.get(symbol.upper(), symbol.lower())
        data = self._coinglass_get(f'/etf/{coin}/list').get('data') or []
        lookup = {}
        for r in data:
            tk = self._safe_str(r.get('ticker')).upper()
            if not tk:
                continue
            det = r.get('asset_details') or {}
            lookup[tk] = {
                'fund_name': self._safe_str(r.get('fund_name')),
                'fund_type': self._safe_str(r.get('fund_type')),
                'region': self._safe_str(r.get('region')),
                'primary_exchange': self._safe_str(r.get('primary_exchange')),
                'list_date': self._safe_str(r.get('list_date')),
                'management_fee_percent': self._safe_float(r.get('management_fee_percent')),
                'aum_usd': self._safe_float(r.get('aum_usd')),
                'etf_price_usd': self._safe_float(r.get('price_usd')),
                'volume_usd': self._safe_float(r.get('volume_usd')),
                'holding_quantity': self._safe_float(det.get('holding_quantity')),
                'nav_usd': self._safe_float(det.get('net_asset_value_usd')),
                'premium_discount_percent':
                    self._safe_float(det.get('premium_discount_percent')),
            }
        return lookup

    def _fetch_etf_flows_raw(self, symbol: str) -> list:
        """获取 ETF 资金流时序（JOIN 左表）。API: GET /etf/{coin}/flow-history"""
        coin = self.CG_SYMBOL_MAP.get(symbol.upper(), symbol.lower())
        return self._coinglass_get(f'/etf/{coin}/flow-history').get('data') or []

    def fetch_etf_flows(self, assetList=None, start_time=None, end_time=None,
                        save_to_csv: bool = True) -> pd.DataFrame:
        """
        ⑨ 获取 ETF 资金流 × 元数据（客户端 JOIN 后写扁平宽表）

        【数据源】 GET /etf/{coin}/list          → 全量 ETF 快照（元数据）
                 GET /etf/{coin}/flow-history  → 资金流分解（时序）
                 - 需 CG_API_KEY (Hobbyist ✅)，单币种 2 次调用
                 - 覆盖: BTC(~20 只) / ETH(~12 只)；SOL/XRP 端点目前 404
                 - 更新频率: 日频，美股收盘后 1-2 小时（北京 08:00~12:00 陆续就绪）
                 - 历史范围: ⭐ 全量历史（BTC ETF 2024-01-11 至今）

        【指标含义】 一句话: "Wall Street 通过 ETF 配置加密资产的边际资金面"
                    正净流入 = 机构加仓（利好）/ 负净流入 = 减仓（利空）

        【返回字段】 date / symbol / etf_ticker / flow_usd / day_total_flow
                    + 元数据列（fund_name/region/aum_usd/nav_usd/...）

        【阈值解读】 单 ETF 单日 flow_usd（BTC 口径）:
                    >+$100M 强买入 / <-$100M 大幅流出；
                    premium_discount_percent 超出 ±0.5% 为显著异常

        【如何使用】 ⚠️ List 是「当前快照」非历史: 静态字段(fund_name/region/list_date)
                   可安全回溯；动态字段(aum_usd/nav_usd/premium_discount_percent)
                   在所有历史行均为同一份当前值，**仅供参考，勿用于历史回测**。

        Args:
            assetList: 币种列表，默认 ['BTC', 'ETH']
            start_time / end_time / save_to_csv
        Returns:
            pd.DataFrame
        """
        assets = assetList or ['BTC', 'ETH']
        out = []
        for sym in assets:
            log(f'⑨ 拉取 {sym} ETF 资金流 × 元数据', 'step')
            try:
                lookup = self._fetch_etf_list(sym)
                flows = self._fetch_etf_flows_raw(sym)
            except Exception as e:
                # 🔴 套餐/鉴权类错误必须冒泡，不能吞成「该币种没有 ETF」——
                #    实测这会让 --all 把 401 报成「✓ 0 行」的假阳性，
                #    掩盖掉「整个 key 没有任何端点权限」这个真问题。
                if any(k in str(e).lower() for k in
                       ('upgrade plan', 'code=401', 'api key', '403')):
                    raise
                log(f'  {sym} 失败（该币种可能尚无现货 ETF）: {str(e)[:80]}', 'warn')
                continue
            if not flows:
                log(f'  {sym} 资金流为空', 'warn')
                continue

            for day in flows:
                ts = int(self._safe_float(day.get('timestamp') or day.get('time')))
                if ts > 1e11:
                    ts //= 1000                    # 毫秒 → 秒
                date = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') if ts else ''
                day_total = self._safe_float(day.get('flow_usd')
                                             or day.get('total_flow_usd'))
                for item in (day.get('etf_flows') or []):
                    tk = self._safe_str(item.get('etf_ticker')).upper()
                    row = {'date': date, 'symbol': sym.upper(), 'etf_ticker': tk,
                           'flow_usd': self._safe_float(item.get('flow_usd')),
                           'day_total_flow': day_total,
                           'unix_timestamp': ts, 'update_time': self._now_str()}
                    row.update(lookup.get(tk, {}))   # JOIN 元数据
                    out.append(row)
            log(f'  {sym}: {len(lookup)} 只 ETF × {len(flows)} 天', 'ok')

        if not out:
            log('未取得任何 ETF 数据', 'warn')
            return pd.DataFrame()

        df = (pd.DataFrame(out)
              .drop_duplicates(subset=['date', 'symbol', 'etf_ticker'], keep='last')
              .sort_values(['symbol', 'date', 'etf_ticker']).reset_index(drop=True))
        df = self._filter_by_date_str(df, start_time, end_time)
        log(f'  合计 {len(df)} 行（1 行 = 1 天 × 1 ETF）', 'ok')

        if save_to_csv:
            for sym in df['symbol'].unique():
                sub = df[df['symbol'] == sym]
                self._save_csv(sub, f'coinglass_etf_flows_{sym}.csv',
                               note=(f'CoinGlass {sym} ETF 资金流 × 元数据（全量历史）\n'
                                     '⚠️ 元数据列是当前快照，勿用于历史回测'))
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

    def validate_chain_tx(self, df: pd.DataFrame,
                          context: str = '交易所链上转账') -> bool:
        """
        校验 ① 的产出

        检查项: 非空 / 必需列齐全 / 地址格式合法且无重复 / 金额非负 /
               时间戳合理 / ⚠️单一交易所占比过高（采样偏差）
        """
        issues, warns = [], []
        need = ['address', 'n_withdrawals', 'total_usd', 'exchanges',
                'first_ts', 'last_ts']

        if df is None or df.empty:
            issues.append('结果为空')
            return self._emit(context, issues, warns)
        miss = [c for c in need if c not in df.columns]
        if miss:
            issues.append(f'缺列: {miss}')
            return self._emit(context, issues, warns)

        bad = df[~df['address'].astype(str).str.match(self.ADDR_RE)]
        if len(bad):
            issues.append(f'{len(bad)} 个地址格式非法')
        dup = int(df['address'].duplicated().sum())
        if dup:
            issues.append(f'{dup} 个地址重复（聚合逻辑有误）')
        if (df['total_usd'] < 0).any():
            issues.append('存在负金额')
        if (df['n_withdrawals'] < 1).any():
            issues.append('存在提币笔数 < 1 的行')

        now = int(time.time())
        ts = df['last_ts'].dropna()
        if len(ts):
            if (ts > now + 86400).any():
                issues.append('存在未来时间戳')
            if (ts < 1577836800).any():          # 2020-01-01
                warns.append('存在 2020 年之前的时间戳，请核实口径')

        top = df['exchanges'].value_counts(normalize=True)
        if len(top) and top.iloc[0] > 0.8:
            warns.append(f'单一交易所占比 {top.iloc[0]:.0%}（{top.index[0]}）'
                         f'—— 采样偏差，建议多拉几个 symbol 摊平')
        warns.append('🔴 提醒：CEX 提现仅为弱先验，必须交叉验证 + 跨链活跃度过滤')
        return self._emit(context, issues, warns)

    def validate_exchange_balance(self, df: pd.DataFrame,
                                  context: str = '交易所余额') -> bool:
        """校验 ② 的产出: 非空 + 有交易所标识列。"""
        issues, warns = [], []
        if df is None or df.empty:
            issues.append('结果为空')
            return self._emit(context, issues, warns)
        if not any('exchange' in c.lower() for c in df.columns):
            warns.append(f'未找到交易所标识列，实际列: {list(df.columns)[:8]}')
        return self._emit(context, issues, warns)

    def validate_supported_coins(self, df: pd.DataFrame,
                                 context: str = '现货币种') -> bool:
        """校验 ③ 的产出: 非空 + 数量合理 + RWA 标的覆盖提示。"""
        issues, warns = [], []
        if df is None or df.empty:
            issues.append('结果为空')
            return self._emit(context, issues, warns)
        if len(df) < 50:
            warns.append(f'仅 {len(df)} 个币种，明显偏少，请核实端点')
        hit = df[df['is_rwa_target']]['symbol'].tolist() if 'is_rwa_target' in df else []
        warns.append(f'RWA 标的覆盖: {hit}' if hit
                     else f'未覆盖任何 RWA 标的 {self.RWA_SYMBOLS}（课题二会用到）')
        return self._emit(context, issues, warns)

    def validate_daily_series(self, df: pd.DataFrame, value_col: str,
                              context: str = '日频序列',
                              expect_min_rows: int = 100,
                              max_gap_days: int = 7) -> bool:
        """
        校验 ④⑤⑥⑦ 这类日频时序的产出（共用）

        检查项: 非空 / 有 date 与数值列 / 日期无重复 / 数值无全空 /
               日期连续性（缺口 > max_gap_days 告警）/ 行数是否明显偏少

        Args:
            df:              待校验 DataFrame
            value_col:       数值列名（如 'fng_value'）
            context:         日志上下文名
            expect_min_rows: 行数下限，低于此值告警
            max_gap_days:    允许的最大日期缺口
        Returns:
            bool
        """
        issues, warns = [], []
        if df is None or df.empty:
            issues.append('结果为空')
            return self._emit(context, issues, warns)
        for c in ('date', value_col):
            if c not in df.columns:
                issues.append(f'缺列: {c}')
        if issues:
            return self._emit(context, issues, warns)

        dup = int(df['date'].duplicated().sum())
        if dup:
            issues.append(f'{dup} 个日期重复')
        if df[value_col].isna().all():
            issues.append(f'{value_col} 全为空')

        d = pd.to_datetime(df['date'], errors='coerce').dropna().sort_values()
        if len(d) > 1:
            gaps = d.diff().dt.days.dropna()
            big = gaps[gaps > max_gap_days]
            if len(big):
                warns.append(f'{len(big)} 处日期缺口 > {max_gap_days} 天，'
                             f'最大 {int(big.max())} 天')
            if d.max() < pd.Timestamp.now() - pd.Timedelta(days=3):
                warns.append(f'最新日期 {d.max().date()} 距今超过 3 天，数据可能滞后')
        if len(df) < expect_min_rows:
            warns.append(f'仅 {len(df)} 行，少于预期 {expect_min_rows} 行')
        return self._emit(context, issues, warns)


# ================================================================
#  CoinglassFetcher — 适配 BaseFetcher，供 FetchOrchestrator 统一编排
# ================================================================
class CoinglassFetcher(BaseFetcher):
    """
    薄适配层：把 CoinGlassDataSource 接进项目的 Fetcher 体系。

    数据逻辑全在 CoinGlassDataSource；本类只负责参数透传与产出对接，
    这样 fetch_all.py 可以和其他 Fetcher 一视同仁地编排它。
    """

    name = 'coinglass'
    desc = 'CoinGlass 数据源（CEX 提币 / 市场环境变量）'
    quality = '★★ 弱先验'
    needs_key = True
    fields = ['address', 'n_withdrawals', 'total_usd', 'exchanges',
              'first_ts', 'last_ts']
    note = ('人类样本候选 —— CEX 提币接收方\n'
            '🔴 CEX 提现只是弱先验，不是证明。必须与 ENS/Farcaster/昼夜节律\n'
            '   交叉验证，并做标签噪声敏感性分析')

    # --source 可选值 → (方法名, 校验方式)
    SOURCES = {
        'chain_tx':   '① 交易所链上转账 → 人类样本候选',
        'balance':    '② 交易所余额快照',
        'coins':      '③ 现货支持币种',
        'fear_greed': '④ 恐慌&贪婪指数',
        'ahr999':     '⑤ AHR999 抄底指标',
        'stablecoin': '⑥ 稳定币总市值',
        'premium':    '⑦ Coinbase 溢价指数',
        'peak':       '⑧ 牛市顶部指标',
        'etf':        '⑨ ETF 资金流 × 元数据',
    }

    def __init__(self, source='chain_tx', all_sources=False, symbol='USDT', pages=20,
                 min_usd=100.0, max_usd=2_000_000.0, interval='1d',
                 assets='BTC,ETH', start_time=None, end_time=None,
                 append=False, probe=False, **kw):
        super().__init__(**kw)
        self.source = source
        self.symbol = symbol
        self.pages = int(pages or 20)
        self.min_usd = float(min_usd or 0)
        self.max_usd = float(max_usd or 1e18)
        self.interval = interval
        self.assets = [a.strip().upper() for a in str(assets).split(',') if a.strip()]
        self.start_time = start_time
        self.end_time = end_time
        self.append = bool(append)
        self.do_probe = bool(probe)
        self.all_sources = bool(all_sources)
        self.ds = CoinGlassDataSource()
        self.df = pd.DataFrame()

    def fetch(self) -> list:
        ds = self.ds
        if self.do_probe:
            ds.probe()
            return []

        s = self.source
        if s == 'chain_tx':
            self.df = ds.fetch_exchange_chain_tx(
                symbol=self.symbol, pages=self.pages,
                start_time=self.start_time, end_time=self.end_time,
                min_usd=self.min_usd, max_usd=self.max_usd,
                save_to_csv=True, append=self.append)
            ds.validate_chain_tx(self.df)
        elif s == 'balance':
            self.df = ds.fetch_exchange_balance(self.symbol)
            ds.validate_exchange_balance(self.df)
        elif s == 'coins':
            self.df = ds.fetch_supported_coins()
            ds.validate_supported_coins(self.df)
        elif s == 'fear_greed':
            self.df = ds.fetch_fear_greed(self.start_time, self.end_time)
            ds.validate_daily_series(self.df, 'fng_value', '恐慌贪婪指数', 1000)
        elif s == 'ahr999':
            self.df = ds.fetch_ahr999(self.start_time, self.end_time)
            ds.validate_daily_series(self.df, 'ahr999_value', 'AHR999', 1000)
        elif s == 'stablecoin':
            self.df = ds.fetch_stablecoin_marketcap(self.start_time, self.end_time)
            ds.validate_daily_series(self.df, 'stablecoin_mcap_usd', '稳定币总市值', 1000)
        elif s == 'premium':
            self.df = ds.fetch_coinbase_premium(self.interval,
                                                self.start_time, self.end_time)
            ds.validate_daily_series(self.df, 'premium_rate',
                                     f'Coinbase溢价({self.interval})', 100)
        elif s == 'peak':
            self.df = ds.fetch_bull_market_peak()
        elif s == 'etf':
            self.df = ds.fetch_etf_flows(self.assets, self.start_time, self.end_time)
        else:
            raise SystemExit(f'未知数据源 {s}，可选: {list(self.SOURCES)}')

        self.stats['API 调用次数'] = ds.n_calls
        self.stats['产出行数'] = len(self.df)
        return []      # 各方法已自行落盘

    # 套餐权限不足的特征串 —— CoinGlass 对档位外的端点返回
    # code=401 / msg="Upgrade plan"，这不是代码错误，是订阅档位限制
    PLAN_DENIED = ('upgrade plan', 'code=401')

    def _is_plan_denied(self, err: Exception) -> bool:
        """判断异常是否为"套餐档位不含此端点"（而非真的出错）。"""
        s = str(err).lower()
        return any(k in s for k in self.PLAN_DENIED)

    def fetch_all_sources(self) -> dict:
        """
        ★ 依次跑全部 9 个数据源，单个失败不影响其余

        【为什么需要】 fetch() 一次只跑一个 source。run_all.sh 里若直接调用，
                     第一个端点（chain_tx）遇到档位限制 401 就整个进程退出，
                     后面 8 个端点根本没机会跑 —— 实测 Hobbyist 档位正是如此。

        【降级规则】
          · code=401 / "Upgrade plan" → 本档位不含该端点，记 'plan_denied'，继续下一个
          · 其他异常                  → 记 'error'，继续下一个
          · 成功                      → 记行数

        Returns:
            dict: {source: 结果说明}
        """
        results = {}
        for src in self.SOURCES:
            self.source = src
            self.df = pd.DataFrame()
            log(f'── {self.SOURCES[src]}', 'step')
            try:
                self.fetch()
                results[src] = f'✓ {len(self.df)} 行'
                log(f'   {results[src]}', 'ok')
            except Exception as e:
                if self._is_plan_denied(e):
                    results[src] = '⏭ 档位不含（需升级套餐）'
                    log(f'   {results[src]}', 'warn')
                else:
                    results[src] = f'✗ {type(e).__name__}: {str(e)[:80]}'
                    log(f'   {results[src]}', 'err')
        return results

    def run(self, save: bool = True):
        # --all：逐个跑完 9 个源，单个失败降级不中断
        if getattr(self, 'all_sources', False):
            log(f'[{self.name}] 全部 {len(self.SOURCES)} 个数据源（逐个降级）', 'step')
            res = self.fetch_all_sources()
            ok = sum(1 for v in res.values() if v.startswith('✓'))
            denied = sum(1 for v in res.values() if v.startswith('⏭'))
            log('', 'info')
            log(f'汇总: {ok} 个成功 / {denied} 个档位不含 / '
                f'{len(res) - ok - denied} 个出错', 'ok')
            for k, v in res.items():
                log(f'  {self.SOURCES[k]:<28} {v}')
            if denied == len(res):
                # 全部被拒 —— 不是「某些高级端点要升级」，而是这个 key
                # 当前没有任何端点权限。以下四项已逐一实测排除代码侧原因。
                log('🔴 全部 %d 个端点都返回 401 "Upgrade plan"。' % denied, 'err')
                log('   已实测排除的代码侧原因:', 'info')
                log('     ① key 未被识别? 否 —— 不带 key 返回 "API key missing"，'
                    '带 key 返回 "Upgrade plan"，两者不同即说明 key 被读到了')
                log('     ② header 名写错? 否 —— 试过 6 种，只有 CG-API-KEY '
                    '被识别，其余均返回 "API key missing"')
                log('     ③ API 版本不对? 否 —— v3 与 v4 返回同样的 "Upgrade plan"')
                log('     ④ 端点路径写错? 否 —— 不存在的路径返回 404 '
                    '"Endpoint not found"，真实端点返回 401，'
                    '说明服务端先路由后鉴权，401 是真实的权限判定')
                log('   ⇒ key 有效、调用方式正确，但该账户当前无任何端点权限。', 'warn')
                log('   请到 coinglass.com 确认（这三项只能你自己查）:', 'info')
                log('     · 订阅是否已生效、是否已过期')
                log('     · 这个 key 是否属于已订阅的那个账户'
                    '（同一邮箱下可能有多个 key）')
                log('     · 所购套餐是否覆盖 Open API（部分套餐只含网页端功能）')
            elif denied:
                log(f'🔴 {denied} 个端点返回 401 "Upgrade plan" —— '
                    f'当前档位（config.COIN_GLASS_TIER）不含这些端点，'
                    f'不是代码问题', 'warn')
            return None

        log(f'[{self.name}] {self.SOURCES.get(self.source, self.desc)}', 'step')
        self.fetch()
        for k, v in self.stats.items():
            log(f'  {k}: {v}')
        return None

    @classmethod
    def add_args(cls, ap):
        ap.add_argument('--source', default='chain_tx', choices=list(cls.SOURCES),
                        help='选择数据源：' +
                             ' / '.join(f'{k}={v}' for k, v in cls.SOURCES.items()))
        ap.add_argument('--symbol', default='USDT', help='①② 用：代币符号')
        ap.add_argument('--pages', type=int, default=20, help='① 用：翻页数')
        ap.add_argument('--min-usd', type=float, default=100.0, help='① 用：金额下限')
        ap.add_argument('--max-usd', type=float, default=2_000_000.0,
                        help='① 用：金额上限')
        ap.add_argument('--interval', default='1d', choices=['1d', '4h', '1h'],
                        help='⑦ 用：K 线粒度（1h 需 Standard 起）')
        ap.add_argument('--assets', default='BTC,ETH', help='⑨ 用：币种，逗号分隔')
        ap.add_argument('--start-time', default=None, help='起始 YYYY-MM-DD')
        ap.add_argument('--end-time', default=None, help='结束 YYYY-MM-DD')
        ap.add_argument('--append', action='store_true', help='① 用：追加而非覆盖')
        ap.add_argument('--all', dest='all_sources', action='store_true',
                        help='★ 依次跑全部 9 个数据源，单个档位不足/出错自动跳过')
        ap.add_argument('--probe', action='store_true',
                        help='探测本档位能访问哪些端点（9 个，约 22 秒）')
        return ap


def main():
    ap = argparse.ArgumentParser(
        description='CoinGlass Open API v4 数据采集（9 个数据源）')
    CoinglassFetcher.add_args(ap)
    args = ap.parse_args()
    CoinglassFetcher(**{k: v for k, v in vars(args).items() if v is not None}).run()


if __name__ == '__main__':
    main()
