#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：make_synthetic.py
@Description:
    合成数据生成器 —— 只为验证管线，不产生任何科学结论。

    ============================================================
    一、 ⚠️⚠️ 重要声明
    ============================================================
      本脚本按 H1 假设「造」出三组可分的数据。
      **跑出 GO 是必然的，因为答案是我们自己写进去的。**

      它唯一的用途: 在没有 API key、没等到真实数据之前，
      5 分钟内确认 fetch → features → modeling 这条管线是通的。

      🔴 真实判定必须用 src/data_fetch/fetch_txs.py 拉链上数据。

    ============================================================
    二、 三组的行为参数（按 H1 构造）
    ============================================================
      组      Δt 对数正态 μ    突发概率   夜间活动权重   gas 恒定概率
      bot     ln(1.2)         0.05       1.00          0.92
      agent   ln(4.5)         0.45       0.95          0.35
      human   ln(900)         0.12       0.18          0.05

      ※ --overlap 参数可把三组分布拉近（0=清晰可分 / 1=完全重叠），
        用于测试 go/no-go 判定逻辑的两端行为。

    ============================================================
    三、 关键实现细节
    ============================================================
      1. ⚠️ 16**40 会溢出 int64 ——
         随机地址必须按字节生成（rand_hex），不能用 rng.integers(0, 16**40)
      2. 时间戳按累积间隔生成后取整到秒，与真实链上的整秒时间戳一致
      3. human 组的夜间（UTC 0-6）活动被抽稀，模拟昼夜节律

    ============================================================
    四、 本文件在课题中的定位
    ============================================================
      🧪 **烟测工具** —— 产出落到 data/raw/synthetic/，
        供 compute_features --synthetic 与 plot_gonogo --synthetic 使用。
        判定书会自动加醒目免责声明。
'''

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config
from src.utils.common import log

# 每组的行为参数：(Δt 对数正态的 μ, σ, burst 概率, 夜间活动权重, gas 固定概率)
PROFILES = {
    "bot": dict(log_mu=np.log(1.2), log_sigma=0.55, burst_p=0.05,
                night_w=1.0, gas_fixed_p=0.92, n_tx=(200, 900)),
    "agent": dict(log_mu=np.log(4.5), log_sigma=0.75, burst_p=0.45,
                  night_w=0.95, gas_fixed_p=0.35, n_tx=(120, 600)),
    "human": dict(log_mu=np.log(900), log_sigma=1.6, burst_p=0.12,
                  night_w=0.18, gas_fixed_p=0.05, n_tx=(30, 180)),
}

START_TS = 1_756_000_000  # 约 2025-08
BLOCK_T = 2.0             # Base 出块 2s


def rand_hex(rng: np.random.Generator, n_bytes: int) -> str:
    """生成 0x 开头的随机十六进制串（16**40 会溢出 int64，必须按字节生成）。"""
    return "0x" + rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes().hex()


def lerp(a: float, b: float, w: float) -> float:
    return a * (1 - w) + b * w


def kind_of(group: str) -> str:
    """
    模拟三组的活动类型构成（对齐 fetch_activity 的实测发现）

    实测: MEV bot 几乎全是 tx_from；x402 付款方几乎全是 event；
          Olas Safe 主要是 tx_to。
    """
    return {"bot": "tx_from", "human": "tx_from", "agent": "event"}.get(group, "tx_from")


def gen_address(rng: np.random.Generator, group: str, overlap: float) -> list:
    """
    overlap ∈ [0,1] 把「全部」行为参数向公共均值插值：
    0 = 按 H1 清晰可分（管线冒烟测试应得 GO）
    1 = 三组完全同分布（判定逻辑测试应得 NO-GO）
    ⚠️ 若只模糊延迟参数，分类器会靠 gas/交互结构照样分对 —— 测不到 NO-GO 分支。
    """
    p = PROFILES[group]
    # n_tx 也要插值：样本量会透过 unique_to_ratio / method_entropy 等比值特征泄漏组别
    lo = int(lerp(p["n_tx"][0], 100, overlap))
    hi = int(lerp(p["n_tx"][1], 500, overlap))
    n = int(rng.integers(lo, max(hi, lo + 1)))

    # 公共均值（overlap=1 时三组全部参数退化到同一组值）
    mu = lerp(p["log_mu"], np.log(30), overlap)
    sigma = lerp(p["log_sigma"], 1.9, overlap)   # ⚠️ 必须插值而非加法，否则尾宽仍可分
    burst_p = lerp(p["burst_p"], 0.2, overlap)
    night_w = lerp(p["night_w"], 0.7, overlap)
    gas_fixed_p = lerp(p["gas_fixed_p"], 0.45, overlap)

    dt = rng.lognormal(mu, sigma, n)
    # 突发簇：一部分间隔被压到很短，模拟事件驱动的连发
    burst = rng.random(n) < burst_p
    dt[burst] *= rng.uniform(0.05, 0.25, burst.sum())

    ts = START_TS + np.cumsum(np.concatenate([[0], dt]))[:n]

    # 昼夜节律：human 夜间(UTC 0-6)活动被抽稀
    hours = ((ts // 3600) % 24).astype(int)
    keep = np.ones(n, dtype=bool)
    night = (hours >= 0) & (hours < 6)
    keep[night] = rng.random(night.sum()) < night_w
    ts = ts[keep]
    if len(ts) < config.MIN_TX_FOR_FEATURES + 5:
        ts = START_TS + np.cumsum(rng.lognormal(mu, sigma, 60))

    ts = np.sort(ts)
    blocks = ((ts - START_TS) / BLOCK_T).astype(int) + 20_000_000

    fixed_gas = int(rng.integers(5e7, 5e8))
    n_targets = int(round(lerp({"bot": 3, "agent": 12, "human": 25}[group], 13, overlap)))
    targets = [rand_hex(rng, 20) for _ in range(max(n_targets, 1))]
    n_methods = int(round(lerp({"bot": 2, "agent": 6, "human": 12}[group], 7, overlap)))
    methods = [rand_hex(rng, 4) for _ in range(max(n_methods, 1))]

    span_s = float(ts[-1] - ts[0]) if len(ts) > 1 else 1.0
    out = []
    for i, (t, b) in enumerate(zip(ts, blocks)):
        gas = fixed_gas if rng.random() < gas_fixed_p else int(rng.integers(3e7, 9e8))
        out.append({
            "hash": rand_hex(rng, 32),
            "blockNumber": int(b),
            "timeStamp": int(t),
            "to": str(rng.choice(targets)),
            "value": "0" if rng.random() < 0.7 else str(int(rng.integers(1, 100)) * 10**16),
            "gasPrice": str(gas),
            "gasUsed": str(int(rng.integers(21000, 300000))),
            "nonce": i,
            "isError": "1" if rng.random() < lerp(0.08 if group == "bot" else 0.02, 0.04, overlap) else "0",
            "methodId": str(rng.choice(methods)),
            "txIndex": int(rng.integers(0, 60)),
            # ★ 统一活动流的两个新字段（与 fetch_activity 产出对齐）
            "kind": kind_of(group),
            "seg": int((t - START_TS) / max(span_s / 12, 1)),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="生成合成数据以验证管线")
    ap.add_argument("--n", type=int, default=80, help="每组地址数")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="0=三组清晰可分(GO)，1=完全重叠(NO-GO)。用来测试判定逻辑两端")
    args = ap.parse_args()

    base = config.DATA_RAW / "synthetic"
    if base.exists():
        shutil.rmtree(base)

    log(f"生成合成数据：每组 {args.n} 个地址，overlap={args.overlap}", "step")
    log("⚠️ 这是假数据，结论无效，仅用于验证管线", "warn")

    rng = np.random.default_rng(config.RANDOM_SEED)
    for group in config.GROUPS:
        d = base / group
        d.mkdir(parents=True, exist_ok=True)
        for _ in range(args.n):
            addr = rand_hex(rng, 20)
            (d / f"{addr}.json").write_text(json.dumps(gen_address(rng, group, args.overlap)))
        log(f"[{group}] {args.n} 个地址", "ok")

    log(f"写入 {base}", "ok")
    log("下一步：python src/compute_features.py --synthetic", "info")


if __name__ == "__main__":
    main()
