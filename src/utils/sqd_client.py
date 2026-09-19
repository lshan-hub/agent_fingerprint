#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：sqd_client.py
@Description:
    SQD Network 公开 Portal 客户端 —— 主力取数通道，免费全历史。

    ============================================================
    一、 认证与限频
    ============================================================
      Base URL: https://portal.sqd.dev
      认证:     无 —— 无需 API Key、无需注册、零成本
      数据集:   139 个，全部 start_block=0（全历史）
      限频:     无硬限流；实测 8 并发最优 5.22 req/s（24/24 成功）
                16 并发降到 3.09 req/s 且 32/48 失败
      限流信号: HTTP 529（偶发 429/503），指数退避后实测 100% 恢复

    ============================================================
    二、 三个端点
    ============================================================
      GET  /datasets/{ds}/metadata   数据集元信息（含 start_block）
      GET  /datasets/{ds}/head       当前链头高度
      POST /datasets/{ds}/stream     NDJSON 流，每行一个区块对象

    ============================================================
    三、 实测踩到的四个坑（本模块已全部处理）
    ============================================================
      1. ⚠️ Python urllib 默认 UA 被 403 拦截 —— 必须显式设置 User-Agent
         （同一请求 curl 返回 200，urllib 返回 403，看起来像权限问题，最耗时）
      2. ⚠️ 响应是 NDJSON 流（每行一个区块），不是单个 JSON 对象
         json.loads(response) 会直接报错，必须逐行解析
      3. ★ 每次响应只推进约 396 个区块，**与过滤器无关**。
         稀疏地址和稠密地址请求同一范围，都停在同一个区块。
         → 因此「按地址逐个查」是错的架构，
           必须「一次全扫描 + 过滤器塞进全部地址（上限 1000 个）」
      4. ⚠️ 不带数据过滤器时按每 100 个区块**采样**返回 ——
         据此算出块时间会得到正好 100 倍的错误值。
         测量类查询必须加 "transactions":[{}] 强制返回每一块。

    ============================================================
    四、 字段归一化
    ============================================================
      SQD 返回十六进制（'0x0'），Etherscan 返回十进制（'0'）；
      且 status(1=成功) 与 isError(0=成功) **语义相反**。
      _normalize 已统一成 Etherscan 口径，下游无需分辨来源。

    ============================================================
    五、 本文件在课题中的定位
    ============================================================
      🥇 **主力取数通道** —— x402 / MEV 日志扫描、大批量地址交易史、
        RQ3 全网扫描（1586 万区块 / 2.1 小时 / 95GB Parquet）都走它。
      🔴 单点风险: SQD 是免费公共品，无 SLA。
        第一周的头号任务不是建模，是把数据拉下来落盘。
'''

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils.common import log  # noqa: E402


class SQDClient:
    # 🔴 「稍后重试就好」的瞬时错误码，与网络中断同类，共用 SQD_NET_RETRIES 预算。
    #   429/503/529 = SQD 自身限流与过载
    #   500/502/504 = 上游网关瞬时故障
    #   520~524     = Cloudflare 故障族（520 Unknown / 522 Conn Timeout / 524 Timeout）
    #   实测漏掉 520 时，human 采集阶段3 一次就抛出中断 ——
    #   它走 HTTPError 分支，此前不吃网络预算，重试完 6 次就死。
    TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504,
                                 520, 521, 522, 523, 524, 529})

    # ============================================================
    # 🔴 自适应节流 —— 2026-09-05 实测新增
    # ============================================================
    # 【问题】 每个请求独立退避，彼此没有记忆。SQD 进入限流期后，
    #         每个新请求都从 1s 重新开始爬坡，既慢又持续给服务端加压。
    #         实测 MEV 采集时几乎每个请求都撞 529，重试 10 次（累计 9 分钟）
    #         仍失败，整个数据源被判软失败。
    #
    # 【做法】 全进程共享一个「请求间隔」:
    #           撞 529  → 间隔 ×2 + 0.5s（乘性增加：快速让路）
    #           连续成功 → 间隔 −0.25s   （加性减少：缓慢提速）
    #         与 TCP 拥塞控制同一思路，只是方向相反 ——
    #         TCP 调的是"发送速率"（加性增、乘性减），
    #         这里调的是"请求间隔"，所以变成乘性增、加性减。
    #         乘性增让限流期迅速降压，加性减避免刚恢复就又打满。
    #
    # 【代价】 正常期间隔会衰减到 0，不影响速度；
    #         只在服务端真的过载时才降速，且能自动恢复。
    _throttle = 0.0          # 类变量：同进程内所有实例共享
    # 🔴 3.0 而非 8.0 —— 2026-09-06 实测修正
    #   SQD 的 529 是「服务端整体过载」，不是针对本客户端的速率限流:
    #   实测只发几个请求就撞 529，且持续 6 小时稳定在 40~60% 成功率。
    #   这种情况下把间隔自罚到 8s 并不能提高成功率，只是白等 ——
    #   真正穿透随机失败的是重试预算（10 次，失败率 0.5^10 ≈ 0.1%）。
    #   3s 仍能缓解「自己打太快」造成的限流，又不会在服务端过载时
    #   把采集拖慢到不可用（760 次请求 × 8s = 100 分钟纯等待）。
    THROTTLE_MAX = 3.0
    THROTTLE_STEP = 0.25     # 每次成功后减少的间隔（秒）

    def __init__(self, dataset: str = None, workers: int = None, chain: str = None):
        """
        dataset: SQD 数据集名，如 base-mainnet
        chain:   本项目链名（base/ethereum/bsc/...），会自动映射到数据集名。
                 ⚠️ BSC 映射到 binance-mainnet，不是 bsc-mainnet
        """
        if chain:
            dataset = config.SQD_DATASETS.get(chain, dataset)
            bt = config.SQD_BLOCK_TIME_MEASURED.get(chain)
            if bt and bt >= 10:
                log(f"{chain} 出块 {bt}s ⇒ 「agent 带」(2-10s) 在物理上不存在，"
                    f"H1 无法在此链检验。仅可用于注册表枚举/地址采集。", "warn")
        self.dataset = dataset or config.SQD_DATASET
        self.workers = workers or config.SQD_WORKERS
        self.base = f"{config.SQD_PORTAL}/datasets/{self.dataset}"
        self.headers = {
            "Content-Type": "application/json",
            # ⚠️ 坑 1：不设 UA 会被 403。curl 能通、urllib 不能，极易误判为权限问题
            "User-Agent": config.SQD_USER_AGENT,
        }

    # ---------- 底层请求 ----------
    # 🔴 90 而非 300：urlopen 的 timeout 只约束「单次 socket 操作」，不是总时长。
    #    服务器若持续涓流发送，每次 read 都不超时，整个请求可以挂起无限久 ——
    #    实测遇到过单次 _post 挂起 12 小时。收紧到 90s 让慢连接尽早失败转重试，
    #    正常请求 1~4s 返回，90s 仍有 20 倍余量。
    def _post(self, path: str, body: dict, timeout: int = 90) -> bytes:
        url = f"{self.base}{path}"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=self.headers
        )
        last = None
        self._waited = 0.0
        # 🔴 网络类错误用独立且更大的重试预算。
        #    实测教训: 一次几分钟的网络中断打穿了 6 次重试
        #    （退避 1+2+4+8+16+32 = 63 秒），直接让 MEV 与 human 两组采集失败，
        #    地址清单为空 → activity 阶段跳过 → 最后只剩 agent 一个类别，
        #    7 套模型全部训练失败。上一版把封顶放宽到 120s 是无效的 ——
        #    2^5 = 32 根本够不到封顶，真正的瓶颈是「次数」不是「单次上限」。
        #    改为网络错误最多 SQD_NET_RETRIES(10) 次，
        #    退避 1,2,4,8,16,32,64,120,120,120 ≈ 8 分钟，足够熬过常见抖动。
        max_try = max(config.SQD_MAX_RETRIES, config.SQD_NET_RETRIES)
        n_net = 0
        for attempt in range(max_try):
            try:
                if SQDClient._throttle > 0:      # 限流期主动让路
                    time.sleep(SQDClient._throttle)
                data = urllib.request.urlopen(req, timeout=timeout).read()
                # 成功 ⇒ 加性减少，缓慢提速（8s 满速需约 32 次成功）
                if SQDClient._throttle:
                    SQDClient._throttle = max(
                        SQDClient._throttle - self.THROTTLE_STEP, 0.0)
                return data
            except urllib.error.HTTPError as e:
                last = e
                # 瞬时错误（限流 / 上游网关 / Cloudflare 故障族）与网络中断同类，
                # 共用 SQD_NET_RETRIES 预算；4xx（除 429）是确定性错误，
                # 重试没有意义，直接 raise 让调用方看到真实原因。
                #
                # 实测: 长时间连续扫描（human 组 607 次请求）会稳定触发 529，
                # 封顶 30s 的退避（累计仅 61s）熬不过限流窗口 ⇒ 放宽到 120s，
                # 并加 ±25% 抖动避免与自身请求节奏共振。
                if e.code not in self.TRANSIENT_CODES:
                    raise
                n_net += 1
                # 乘性增加：让后续请求整体降速，而不是每个请求各自硬扛
                SQDClient._throttle = min(
                    SQDClient._throttle * 2 + 0.5, self.THROTTLE_MAX)
                if n_net > config.SQD_NET_RETRIES:
                    break
                cap = 30 if e.code in (429, 503) else 120
                wait = min(2 ** attempt, cap)
                wait *= 0.75 + 0.5 * (hash((url, attempt)) % 1000) / 1000.0
                log(f"SQD {e.code}，第 {n_net}/{config.SQD_NET_RETRIES} 次"
                    f"重试前等 {wait:.0f}s", "warn")
                time.sleep(wait)
                continue
            except Exception as e:
                last = e
                # 🔴 DNS/连接类错误要比一般异常更能熬 —— 实测一次本地 DNS 抖动
                #    （Errno 8 nodename nor servname provided）就打穿了 6 次重试
                #    （累计仅 61s），把跑了 2 小时的采集整个炸掉。
                #    网络类错误封顶放宽到 120s，累计约 4 分钟，足够熬过多数抖动。
                msg = str(e).lower()
                is_net = any(k in msg for k in (
                    'nodename nor servname', 'name resolution', 'name or service',
                    'temporary failure', 'connection reset', 'connection refused',
                    'timed out', 'urlopen error'))
                if not is_net and attempt >= config.SQD_MAX_RETRIES - 1:
                    break            # 非网络错误只用基础预算
                n_net += 1
                if is_net and n_net > config.SQD_NET_RETRIES:
                    break
                wait = min(2 ** attempt, 120 if is_net else 30)
                if is_net:
                    log(f"SQD 网络异常（{type(e).__name__}），第 {n_net}/"
                        f"{config.SQD_NET_RETRIES} 次重试前等 {wait:.0f}s"
                        f"（累计已等 {self._waited:.0f}s）", "warn")
                    self._waited += wait
                time.sleep(wait)
        raise RuntimeError(
            f"SQD 重试失败（网络重试 {n_net}/{config.SQD_NET_RETRIES}）: {last}")

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.base}{path}", headers=self.headers)
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    @staticmethod
    def _parse_ndjson(raw: bytes) -> list:
        """⚠️ 坑 2：响应是 NDJSON —— 每行一个区块对象，不能整体 json.loads。"""
        return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]

    # ---------- 元信息 ----------
    def metadata(self) -> dict:
        return self._get("/metadata")

    def head(self) -> int:
        return self._get("/head")["number"]

    def block_time(self, block: int) -> int:
        raw = self._post("/stream", {
            "type": "evm", "fromBlock": block, "toBlock": block,
            "fields": {"block": {"number": True, "timestamp": True}},
        })
        blocks = self._parse_ndjson(raw)
        return blocks[0]["header"]["timestamp"] if blocks else None

    def find_block_by_date(self, date_str: str) -> int:
        """二分查找某日期对应的区块高度。用于确定建模时间窗起点。"""
        target = time.mktime(time.strptime(date_str, "%Y-%m-%d"))
        lo, hi = 1, self.head()
        for _ in range(config.SQD_BISECT_ROUNDS):
            mid = (lo + hi) // 2
            ts = self.block_time(mid)
            if ts is None:
                lo = mid
            elif ts < target:
                lo = mid
            else:
                hi = mid
        return hi

    # ---------- 核心：扫描 ----------
    def scan_chunk(self, from_block: int, to_block: int,
                   addresses: list = None, fields: dict = None) -> tuple:
        """
        扫描一段区块。返回 (blocks, next_from_block)。

        ⚠️ 坑 3：无论请求多大范围，单次只推进约 SQD_CHUNK_BLOCKS(≈396) 个区块。
        调用方必须用返回的 next_from_block 续拉，直到 >= to_block。
        """
        tx_filter = {}
        if addresses:
            tx_filter["from"] = [a.lower() for a in addresses]

        body = {
            "type": "evm",
            "fromBlock": from_block,
            "toBlock": to_block,
            "fields": fields or config.SQD_FIELDS,
            "transactions": [tx_filter],
        }
        blocks = self._parse_ndjson(self._post("/stream", body))
        if not blocks:
            return [], to_block + 1
        return blocks, blocks[-1]["header"]["number"] + 1

    def stream(self, query: dict, on_progress=None):
        """
        通用查询生成器 —— 自动处理「单次只推进约 396 区块」的分页。

        query 是完整的 SQD 请求体（可含 logs / transactions / traces 任意过滤器），
        其中的 fromBlock / toBlock 会被本方法接管用于续拉。

        用法：
            for blk in client.stream({"type":"evm","fromBlock":a,"toBlock":b,
                                      "fields":{...},"logs":[{...}]}):
                ...

        ⚠️ 这是串行续拉。大范围扫描请用 scan_range（8 并发）。
        """
        body = dict(query)
        cur = int(body["fromBlock"])
        end = int(body["toBlock"])
        n_req = 0
        while cur <= end:
            body["fromBlock"] = cur
            blocks = self._parse_ndjson(self._post("/stream", body))
            n_req += 1
            if not blocks:
                break
            for b in blocks:
                yield b
            nxt = blocks[-1]["header"]["number"] + 1
            if nxt <= cur:          # 防御：没有推进就退出，避免死循环
                break
            cur = nxt
            if on_progress and n_req % 20 == 0:
                on_progress(cur, end, n_req)

    def scan_range(self, from_block: int, to_block: int,
                   addresses: list = None, fields: dict = None,
                   on_progress=None) -> list:
        """
        并发扫描一个区块范围，返回全部命中区块。

        做法：把 [from, to] 切成若干段并发跑，每段内部串行续拉。
        实测 8 并发最优（5.2 req/s）；16 并发吞吐反降且大量 529。
        """
        total = to_block - from_block
        n_seg = max(self.workers, 1)
        seg_size = max(total // n_seg, config.SQD_CHUNK_BLOCKS)
        segments = [
            (from_block + i * seg_size,
             min(from_block + (i + 1) * seg_size - 1, to_block))
            for i in range((total // seg_size) + 1)
        ]
        segments = [(a, b) for a, b in segments if a <= b]

        def run_segment(seg):
            lo, hi = seg
            out, cur = [], lo
            while cur <= hi:
                blocks, cur = self.scan_chunk(cur, hi, addresses, fields)
                out.extend(blocks)
                if not blocks and cur > hi:
                    break
            return out

        results, done = [], 0
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(run_segment, s): s for s in segments}
            for fut in as_completed(futures):
                try:
                    results.extend(fut.result())
                except Exception as e:
                    log(f"分段 {futures[fut]} 失败: {e}", "err")
                done += 1
                if on_progress:
                    on_progress(done, len(segments))
        return results

    # ---------- 面向本课题的封装 ----------
    def fetch_addresses(self, addresses: list, from_block: int, to_block: int,
                        on_progress=None) -> dict:
        """
        批量拉取多个地址的交易史，返回 {address: [tx, ...]}。

        实测 1000 个地址可一次传入过滤器，所以 1200 个真值地址只需 2 批，
        而不是 1200 次查询 —— 这是相对 Etherscan 架构的根本差异。
        """
        out = {a.lower(): [] for a in addresses}
        batches = [
            addresses[i:i + config.SQD_MAX_ADDR_PER_QUERY]
            for i in range(0, len(addresses), config.SQD_MAX_ADDR_PER_QUERY)
        ]
        log(f"共 {len(addresses)} 个地址，分 {len(batches)} 批过滤", "info")

        for bi, batch in enumerate(batches, 1):
            log(f"批次 {bi}/{len(batches)}（{len(batch)} 个地址）", "step")
            blocks = self.scan_range(from_block, to_block, batch,
                                     on_progress=on_progress)
            for b in blocks:
                ts = b["header"]["timestamp"]
                bn = b["header"]["number"]
                for tx in b.get("transactions", []):
                    frm = (tx.get("from") or "").lower()
                    if frm in out:
                        out[frm].append(self._normalize(tx, bn, ts))

        for a in out:
            out[a].sort(key=lambda t: (t["timeStamp"], t.get("txIndex", 0)))
        return out

    @staticmethod
    def _to_dec(v) -> str:
        """
        ⚠️ 坑 4：SQD 的数值字段返回十六进制字符串（'0x0'、'0xb48a'），
        而 Etherscan 返回十进制（'0'、'46218'）。下游 compute_features 直接
        float() 会抛 ValueError。两个后端必须在此统一为十进制字符串。
        """
        if v is None:
            return "0"
        if isinstance(v, int):
            return str(v)
        s = str(v)
        try:
            return str(int(s, 16)) if s.startswith("0x") else str(int(s))
        except ValueError:
            return "0"

    @classmethod
    def _normalize(cls, tx: dict, block_number: int, timestamp: int) -> dict:
        """转成与 Etherscan 后端一致的结构，让下游 compute_features 无需改动。"""
        return {
            "hash": tx.get("hash", ""),
            "blockNumber": block_number,
            "timeStamp": timestamp,
            "to": (tx.get("to") or "").lower(),
            "value": cls._to_dec(tx.get("value")),
            "gasPrice": cls._to_dec(tx.get("gasPrice")),
            "gasUsed": cls._to_dec(tx.get("gasUsed")),
            "nonce": int(cls._to_dec(tx.get("nonce"))),
            # SQD 的 status: 1=成功 0=失败；Etherscan 的 isError 语义相反
            "isError": "0" if tx.get("status", 1) == 1 else "1",
            # ★ sighash 是 method_entropy 特征的唯一来源，全网扫描时必须保留
            "methodId": tx.get("sighash") or "0x",
            "txIndex": int(cls._to_dec(tx.get("transactionIndex"))),
        }


def selftest():
    """连通性自检 —— 开题前和每次长时间未用后都该跑一次。"""
    c = SQDClient()
    log("SQD 连通性自检", "step")

    md = c.metadata()
    log(f"数据集 {md.get('dataset')} | start_block={md.get('start_block')} "
        f"| real_time={md.get('real_time')}", "ok")
    if md.get("start_block") != 0:
        log("start_block 非 0，全历史可能不可用", "warn")

    head = c.head()
    log(f"当前链头 {head:,}", "ok")

    t0 = time.time()
    blocks, nxt = c.scan_chunk(head - 5000, head - 1000)
    adv = nxt - (head - 5000)
    log(f"单次扫描推进 {adv} 区块，耗时 {time.time() - t0:.1f}s "
        f"（实测基准 ≈{config.SQD_CHUNK_BLOCKS}）", "ok")

    start = c.find_block_by_date(config.WINDOW_START_DATE)
    log(f"{config.WINDOW_START_DATE} ≈ 区块 {start:,} "
        f"（实测参考值 {config.WINDOW_START_BLOCK:,}）", "ok")
    log("自检通过", "ok")


if __name__ == "__main__":
    selftest()
