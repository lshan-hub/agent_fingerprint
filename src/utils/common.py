#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：common.py
@Description:
    共用工具 —— Etherscan 限速请求 / 磁盘缓存 / 日志 / 中文绘图字体。

    ============================================================
    一、 提供的能力
    ============================================================
      · log(msg, level)              统一日志输出（info/ok/warn/err/step）
      · etherscan_get(chain, params) Etherscan V2 限速请求（含退避重试）
      · load_cache / save_cache      按 (group, address, chain) 落盘缓存
      · read_address_csv(path)       读三组真值清单（跳过 # 注释）
      · setup_matplotlib_cjk()       配置中文字体与绘图默认样式

    ============================================================
    二、 关键实现细节
    ============================================================
      1. 限速: Etherscan 免费层官方 5 req/s，config.RATE_LIMIT_PER_SEC 默认留到 4.0
      2. ⚠️ Etherscan 的限速错误藏在 HTTP 200 响应里（result 字段含 "rate limit"），
         必须解析响应体才能发现，不能只看状态码
      3. 缓存损坏（JSONDecodeError）时自动删除重拉，避免脏缓存卡住管线
      4. ⚠️ 中文字体（PingFang SC / Hiragino）缺 U+2212 MINUS SIGN 字形，
         已设 axes.unicode_minus=False；轴标签中也应避免直接用 −

    ============================================================
    三、 本文件在课题中的定位
    ============================================================
      🔧 **基础设施** —— 被 data_fetch / feature / modeling 三层共同依赖。
        与 src/utils/_base.py 的分工: _base 面向链上直读与 Fetcher 基类，
        common 面向 Etherscan 与管线通用能力。
'''

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402


# ============================================================
# 日志
# ============================================================
def log(msg: str, level: str = "info") -> None:
    prefix = {
        "info": "  ",
        "ok": "✓ ",
        "warn": "⚠ ",
        "err": "✗ ",
        "step": "▶ ",
    }.get(level, "  ")
    print(f"{prefix}{msg}", flush=True)


# ============================================================
# 限速器 —— Etherscan 免费层 5 req/s
# ============================================================
class RateLimiter:
    def __init__(self, per_sec: float):
        self.interval = 1.0 / per_sec
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last = time.monotonic()


_limiter = RateLimiter(config.RATE_LIMIT_PER_SEC)


# ============================================================
# 带重试的 Etherscan V2 请求
# ============================================================
def etherscan_get(chain: str, params: dict) -> dict:
    """
    调用 Etherscan V2 多链接口。

    返回原始 JSON。调用方负责解读 status/message。
    对限速(rate limit)错误做指数退避重试。
    """
    if chain not in config.CHAINS:
        raise ValueError(f"未知链: {chain}。可选: {list(config.CHAINS)}")

    q = dict(params)
    q["chainid"] = config.CHAINS[chain]
    q["apikey"] = config.get_api_key()

    last_err = None
    for attempt in range(config.MAX_RETRIES):
        _limiter.wait()
        try:
            r = requests.get(
                config.ETHERSCAN_V2_BASE, params=q, timeout=config.REQUEST_TIMEOUT
            )
            r.raise_for_status()
            data = r.json()

            # Etherscan 的限速错误藏在 200 响应里
            msg = str(data.get("result", "")) + str(data.get("message", ""))
            if "rate limit" in msg.lower() or "Max calls per sec" in msg:
                wait = 2 ** attempt
                log(f"触发限速，{wait}s 后重试 (第 {attempt + 1} 次)", "warn")
                time.sleep(wait)
                continue

            return data

        except (requests.RequestException, json.JSONDecodeError) as e:
            last_err = e
            wait = 2 ** attempt
            log(f"请求失败 ({e})，{wait}s 后重试", "warn")
            time.sleep(wait)

    raise RuntimeError(f"重试 {config.MAX_RETRIES} 次仍失败: {last_err}")


# ============================================================
# 磁盘缓存 —— 重跑不重复消耗 API 额度
# ============================================================
def cache_path(group: str, address: str, chain: str) -> Path:
    d = config.DATA_RAW / chain / group
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{address.lower()}.json"


def load_cache(group: str, address: str, chain: str):
    p = cache_path(group, address, chain)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            p.unlink()  # 缓存损坏，删掉重拉
    return None


def save_cache(group: str, address: str, chain: str, payload) -> None:
    cache_path(group, address, chain).write_text(
        json.dumps(payload, ensure_ascii=False)
    )


# ============================================================
# 地址清单读写
# ============================================================
def read_address_csv(path: Path) -> list:
    """
    读地址清单 CSV。要求有 address 列；忽略 # 开头的注释行和空行。
    其余列（source / note 等）原样带出，用于后续溯源。
    """
    import csv

    if not path.exists():
        return []

    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []

    for row in csv.DictReader(lines):
        addr = (row.get("address") or "").strip().lower()
        if addr.startswith("0x") and len(addr) == 42:
            row["address"] = addr
            rows.append(row)
    return rows


# ============================================================
# 中文绘图字体（macOS）
# ============================================================
def setup_matplotlib_cjk() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",          # macOS 默认中文
        "Hiragino Sans GB",
        "Arial Unicode MS",
        "Heiti SC",
        "Microsoft YaHei",      # Windows 兜底
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
