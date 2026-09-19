#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：_base.py
@Description:
    data_fetch 共享底座 —— RPC 轮询 / keccak / 落盘 / BaseFetcher 抽象基类。

    ============================================================
    一、 提供的能力
    ============================================================
      · RpcPool(chain)          多端点轮询 + 失败拉黑 + 指数退避
      · selector(sig)           函数签名 → 4 字节选择器
      · topic0(sig)             事件签名 → topic0
      · words / word_to_addr / enc_uint / dec_string   轻量 ABI 编解码
      · write_csv / write_json / read_source_csv       统一落盘（带注释头）
      · http_json(url)          GET JSON（处理压缩响应）
      · BaseFetcher             所有数据源脚本的统一骨架

    ============================================================
    二、 关键实现细节（均来自 2026-09-03 实测）
    ============================================================
      1. ⚠️ 公共 RPC 单点限流严格 —— 实测 mainnet.base.org 在 0.15s 间隔即 429。
         ⇒ 维护可用端点池并轮询，把有效吞吐提上去；端点连续失败 4 次临时拉黑。
      2. ⚠️ agenteconomy.to 等站点**无视 Accept-Encoding 强推 brotli**，
         urllib 不解压会抛 UnicodeDecodeError: invalid start byte 0x81。
         ⇒ http_json 显式请求 identity，并保留 gzip/deflate/brotli 兜底。
      3. keccak 依赖 pycryptodome；缺失时给出可操作的安装提示而非堆栈。
      4. 所有 CSV 产出带 # 注释头，记录来源、时间、口径 —— 便于事后溯源。

    ============================================================
    三、 BaseFetcher 契约
    ============================================================
      子类只需要:
        1. 覆盖类属性（name / desc / output / quality / needs_key / fields / note）
        2. 实现 fetch() 返回行列表
        3. 可选: 覆盖 add_args() 注册专属 CLI 参数

      统一由 run() 驱动: fetch → 落盘 → 返回路径。
      这样 FetchOrchestrator 可以对所有 Fetcher 一视同仁地编排。

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🔧 **基础设施** —— 被 src/data_fetch/ 下全部 9 个 Fetcher 依赖。
        与 src/utils/common.py 的分工: _base 面向链上直读与 Fetcher 基类，
        common 面向 Etherscan 与管线通用能力。
'''

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402

UA = "agent-fingerprint-research/0.1"
OUT_DIR = config.ROOT / "data" / "sources"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 日志
# ============================================================
def log(msg: str, level: str = "info") -> None:
    print({"info": "  ", "ok": "✓ ", "warn": "⚠ ", "err": "✗ ", "step": "▶ "}
          .get(level, "  ") + msg, flush=True)


# ============================================================
# RPC 端点池（2026-09-03 实测可用，连打 10 次无限流）
# ============================================================
RPC_POOL = {
    "base": [
        "https://mainnet.base.org",
        "https://base-rpc.publicnode.com",
        "https://1rpc.io/base",
    ],
    "gnosis": [
        "https://rpc.gnosischain.com",
        "https://gnosis-rpc.publicnode.com",
        "https://gnosis.drpc.org",
        "https://rpc.gnosis.gateway.fm",
    ],
    "ethereum": [
        "https://ethereum-rpc.publicnode.com",
        "https://eth.drpc.org",
        "https://1rpc.io/eth",
    ],
    "optimism": ["https://optimism-rpc.publicnode.com", "https://1rpc.io/op"],
    "arbitrum": ["https://arbitrum-one-rpc.publicnode.com", "https://1rpc.io/arb"],
    "polygon": ["https://polygon-bor-rpc.publicnode.com", "https://1rpc.io/matic"],
    "bsc": ["https://bsc-rpc.publicnode.com", "https://1rpc.io/bnb"],
    "celo": ["https://forno.celo.org"],
    "mode": ["https://mainnet.mode.network"],
}


class RpcPool:
    """轮询 + 失败拉黑 + 指数退避。"""

    def __init__(self, chain: str, min_interval: float = 0.12):
        self.urls = list(RPC_POOL.get(chain, []))
        if not self.urls:
            raise ValueError(f"没有为 {chain} 配置 RPC，见 _base.RPC_POOL")
        self.chain = chain
        self.i = 0
        self.min_interval = min_interval      # 单端点最小间隔
        self._last = {u: 0.0 for u in self.urls}
        self._fail = {u: 0 for u in self.urls}
        self.n_calls = 0

    def _next(self):
        for _ in range(len(self.urls)):
            u = self.urls[self.i % len(self.urls)]
            self.i += 1
            if self._fail[u] < 4:             # 连错 4 次拉黑
                return u
        self._fail = {u: 0 for u in self.urls}   # 全黑了就重置
        return self.urls[0]

    def call(self, method: str, params: list, retries: int = 4):
        last_err = None
        for attempt in range(retries):
            url = self._next()
            gap = time.monotonic() - self._last[url]
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last[url] = time.monotonic()
            try:
                body = json.dumps({"jsonrpc": "2.0", "id": 1,
                                   "method": method, "params": params}).encode()
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json", "User-Agent": UA})
                data = json.loads(urllib.request.urlopen(req, timeout=30).read())
                if "error" in data:
                    raise RuntimeError(data["error"].get("message", "rpc error"))
                self._fail[url] = 0
                self.n_calls += 1
                return data["result"]
            except Exception as e:
                last_err = e
                self._fail[url] += 1
                if attempt < retries - 1:
                    time.sleep(0.4 * (2 ** attempt))
        raise RuntimeError(f"{method} 重试 {retries} 次失败: {last_err}")


# ============================================================
# keccak / ABI 轻量工具
# ============================================================
def _keccak(text: str) -> str:
    try:
        from Crypto.Hash import keccak as _k
        h = _k.new(digest_bits=256)
        h.update(text.encode())
        return h.hexdigest()
    except ImportError:
        raise SystemExit(
            "\n[缺依赖] 需要 keccak。安装其一：\n"
            "   pip install pycryptodome\n   pip install eth-hash[pycryptodome]\n")


def selector(sig: str) -> str:
    """函数签名 → 4 字节选择器，如 getService(uint256) → 0xef0e239b"""
    return "0x" + _keccak(sig)[:8]


def topic0(sig: str) -> str:
    """事件签名 → topic0"""
    return "0x" + _keccak(sig)


def enc_uint(n: int) -> str:
    return "%064x" % n


def words(hexdata: str) -> list:
    """把 eth_call 返回切成 32 字节 word 列表（去掉 0x）。"""
    h = hexdata[2:] if hexdata.startswith("0x") else hexdata
    return [h[i * 64:(i + 1) * 64] for i in range(len(h) // 64)]


def word_to_addr(w: str) -> str:
    return "0x" + w[24:]


def dec_string(hexdata: str) -> str:
    """解码 ABI 编码的 string（offset+len+data 布局）。"""
    try:
        w = words(hexdata)
        ln = int(w[1], 16)
        raw = "".join(w[2:])[: ln * 2]
        return bytes.fromhex(raw).decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ============================================================
# 落盘
# ============================================================
def write_csv(name: str, rows: list, fieldnames: list, note: str = "") -> Path:
    p = OUT_DIR / name
    with p.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# 由 agent_fingerprint/src/data_fetch 生成\n")
        f.write(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if note:
            for line in note.strip().split("\n"):
                f.write(f"# {line}\n")
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log(f"写入 {p.relative_to(config.ROOT)}（{len(rows)} 行）", "ok")
    return p


def write_json(name: str, obj) -> Path:
    p = OUT_DIR / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    log(f"写入 {p.relative_to(config.ROOT)}", "ok")
    return p


def read_source_csv(name: str) -> list:
    """读回本目录产出的 CSV（跳过 # 注释）。"""
    p = OUT_DIR / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines)) if lines else []


# ============================================================
# HTTP GET（非 RPC 的普通接口）
# ============================================================
# ============================================================
# 🔴 按域名的 HTTP 限频器 —— 2026-09-05 新增
# ============================================================
class DomainThrottle:
    """
    按域名维护「最小请求间隔」，并在撞限流时自适应降速。

    【为什么需要】
      本项目对外有 7 类端点，此前只有 4 类做了限频:
        SQD Portal   ✅ AIMD 自适应节流（sqd_client）
        公共 RPC     ✅ RpcPool 最小间隔 0.12s + 多端点轮询 + 拉黑
        Coinglass    ✅ 按订阅档位的 RPM 节流
        Etherscan    ✅ RateLimiter 4 req/s（common.etherscan_get）
      而 http_json 完全没有限频，它承担了:
        api.dune.com（4 处调用）/ agenteconomy.to / raw.githubusercontent.com
      Dune 免费层与 GitHub raw（未认证 60 次/小时）都有明确限额，
      裸奔迟早撞 429。

    【策略】 与 sqd_client 同一套思路:
        撞 429/503 → 间隔 ×2 + 1s（乘性增加，快速让路）
        连续成功   → 间隔 −0.5s   （加性减少，缓慢提速），不低于基线

    【线程安全】 本项目的 HTTP 调用都是串行的，这里不加锁；
                若将来改成并发，需要给 _state 加 threading.Lock。
    """

    MAX_PENALTY = 30.0      # 单域名惩罚间隔上限（秒）
    PENALTY_STEP = 0.5      # 每次成功后回落的秒数

    def __init__(self):
        self._last = {}       # {域名: 上次请求的 monotonic 时刻}
        self._penalty = {}    # {域名: 当前额外惩罚间隔}

    @staticmethod
    def domain_of(url: str) -> str:
        """从 URL 取域名（不含协议与路径）。"""
        try:
            return url.split('://', 1)[1].split('/', 1)[0].lower()
        except IndexError:
            return url.lower()

    def base_interval(self, domain: str) -> float:
        """该域名的基线最小间隔，取自 config.HTTP_MIN_INTERVAL。"""
        table = getattr(config, 'HTTP_MIN_INTERVAL', {})
        for k, v in table.items():
            if k != '_default' and k in domain:
                return float(v)
        return float(table.get('_default', 0.3))

    def wait(self, url: str) -> None:
        """请求前调用：按基线 + 惩罚间隔补足等待。"""
        d = self.domain_of(url)
        need = self.base_interval(d) + self._penalty.get(d, 0.0)
        gap = time.monotonic() - self._last.get(d, 0.0)
        if gap < need:
            time.sleep(need - gap)
        self._last[d] = time.monotonic()

    def on_success(self, url: str) -> None:
        """成功后调用：惩罚间隔加性回落。"""
        d = self.domain_of(url)
        p = self._penalty.get(d, 0.0)
        if p:
            self._penalty[d] = max(p - self.PENALTY_STEP, 0.0)

    def on_throttled(self, url: str) -> float:
        """撞限流后调用：惩罚间隔乘性增加。Returns: 新的惩罚间隔。"""
        d = self.domain_of(url)
        p = min(self._penalty.get(d, 0.0) * 2 + 1.0, self.MAX_PENALTY)
        self._penalty[d] = p
        return p


# 全模块共享一个实例 —— 同一域名的所有调用共用节流状态
_throttle = DomainThrottle()

# 视为「稍后重试就好」的 HTTP 状态码（与 sqd_client.TRANSIENT_CODES 同口径）
_HTTP_TRANSIENT = frozenset({429, 500, 502, 503, 504,
                             520, 521, 522, 523, 524, 529})


def http_json(url: str, headers: dict = None, timeout: int = 40,
              max_retries: int = 5):
    """
    GET 一个 JSON 接口（自带按域名限频 + 限流退避）。

    ⚠️ 实测坑（agenteconomy.to）：该站无视 Accept-Encoding，
       即使只声明 gzip/deflate 也强推 brotli(br)。
       urllib 不解压，直接 json.loads 会抛
       UnicodeDecodeError: invalid start byte 0x81。

       解法：显式请求 identity（不压缩）。若服务端仍强推 br，
       再尝试用 brotli 库解（Python 3.9 stdlib 没有 brotli）。

    Args:
        url / headers / timeout
        max_retries: 撞 429/5xx 时的重试次数
    Returns:
        dict | list: 解析后的 JSON
    """
    h = {"User-Agent": UA, "Accept": "application/json",
         "Accept-Encoding": "identity"}      # ★ 明确要求不压缩
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)

    last = None
    for attempt in range(max_retries):
        _throttle.wait(url)                  # 🔴 限频：请求前按域名间隔等待
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            raw = resp.read()
            _throttle.on_success(url)
            break
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in _HTTP_TRANSIENT or attempt == max_retries - 1:
                raise                         # 确定性错误：直接抛，重试无意义
            p = _throttle.on_throttled(url)
            wait = min(2 ** attempt, 30)
            log(f"{_throttle.domain_of(url)} 返回 {e.code}，"
                f"第 {attempt+1}/{max_retries} 次重试前等 {wait}s"
                f"（该域名间隔已升到 {p:.1f}s）", "warn")
            time.sleep(wait)
        except Exception as e:
            last = e
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
    else:
        raise RuntimeError(f"http_json 重试 {max_retries} 次仍失败: {last}")

    enc = (resp.headers.get("Content-Encoding") or "").lower()
    if enc == "gzip" or raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)
    elif enc == "deflate":
        import zlib
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    elif enc == "br":
        try:
            import brotli
            raw = brotli.decompress(raw)
        except ImportError:
            raise SystemExit(
                f"\n[缺依赖] {url} 强制返回 brotli 压缩。\n"
                "   pip install brotli\n")

    return json.loads(raw)


# ============================================================
# Fetcher 基类 —— 所有数据源脚本的统一骨架
# ============================================================
class BaseFetcher:
    """
    数据源抓取器的抽象基类。

    子类只需要：
      1. 覆盖类属性（name / output / quality / needs_key …）
      2. 实现 fetch() 返回行列表
      3. 可选：覆盖 add_args() 注册自己的 CLI 参数

    统一由 run() 驱动：fetch → 落盘 → 返回路径。
    这样 fetch_all.py 可以对所有 fetcher 一视同仁地编排。
    """

    # ---- 子类必须/可覆盖的元信息 ----
    name: str = "base"              # 数据源短名（fetch_all 用它做 --skip）
    desc: str = ""                  # 一句话说明
    output: str = ""                # 产出文件名（data/sources/ 下）
    fields: list = ["address", "source", "note"]
    quality: str = ""               # 真值质量，如 "★★★"
    needs_key: bool = False         # 是否需要 API key
    note: str = ""                  # 写进 CSV 注释头的口径说明

    def __init__(self, **opts):
        self.opts = opts
        self.stats = {}             # 子类可往里塞统计量，run() 会打印

    # ---- 子类实现 ----
    def fetch(self) -> list:
        raise NotImplementedError(f"{type(self).__name__} 必须实现 fetch()")

    # ---- 模板方法 ----
    def run(self, save: bool = True):
        log(f"[{self.name}] {self.desc}", "step")
        rows = self.fetch()
        if not rows:
            log(f"[{self.name}] 没有抓到数据", "warn")
            return None
        for k, v in self.stats.items():
            log(f"  {k}: {v}")
        if not save or not self.output:
            return rows
        return write_csv(self.output, rows, self.fields, note=self.note)

    # ---- CLI ----
    @classmethod
    def add_args(cls, ap):
        """子类覆盖以注册专属参数。"""
        return ap

    @classmethod
    def from_args(cls, args):
        """把 argparse 结果转成构造参数。默认把所有非 None 字段透传。"""
        return cls(**{k: v for k, v in vars(args).items() if v is not None})

    @classmethod
    def cli(cls):
        """统一的命令行入口，子类模块里 main() 直接调它即可。"""
        import argparse
        ap = argparse.ArgumentParser(description=cls.desc or cls.__doc__)
        cls.add_args(ap)
        args = ap.parse_args()
        cls.from_args(args).run()
