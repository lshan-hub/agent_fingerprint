#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：sweep_window.py
@Description:
    30k 样本方案的核心采集器 —— 全窗口扫描 + 按活动数预筛 + 配额装配。

    ============================================================
    一、 为什么要有这个模块（取代 fetch_activity 的主路径）
    ============================================================
    旧管线是「先在小切片里发现地址 → 抽 150 个 → 回 7 天窗口按地址查询」，
    四层漏斗相乘只剩 247 个有效样本（详见 config「30k 样本方案」注释）。

    本模块把方向翻转过来：**在建模窗口本身上做发现，发现即采集**，
    并把「有效门槛 ≥20 条」前置成入选条件 —— 入选即有效。

    ============================================================
    二、 六个阶段（CLI 子命令）
    ============================================================
      hotwallets  窗口内均布切片反推 CEX 热钱包（复用 fetch_human 的行为判据）
      main        ★全窗口一趟（logs+transactions 混合选择器，试点已验证可用）:
                    · USDC AuthorizationUsed → x402 付款方注册表 + 事件活动分片
                    · Safe ExecutionSuccess  → Olas Safe 的执行活动分片
                    · 热钱包 tx.from         → 提币接收方（human 候选池）
      mev         隔桶抽样扫 DEX Swap（stride=3 覆盖 1/3 窗口），
                  判据「单桶 arb_tx>=5」与原 5,000 块窗口判据同紧度
      probe       human 候选活跃度探针：48 段均布，按命中数排名取前 14,000
      collect     地址过滤分批采集: bot(36 段) / human(全窗口连续) / olas(全窗口)
      assemble    配额选样 + 卫生规则 + 写 data/raw/{chain}/{group}/*.json
                  + 漏斗报告（论文表 3.1/3.2 的数据来源）

      组合命令: discover = hotwallets+main+mev+probe
               activity = collect+assemble

    ============================================================
    三、 断点续跑（长任务的生存底线）
    ============================================================
      · 每阶段一个状态目录 data/sweep/{阶段}_{lo}_{hi}/（窗口不同互不干扰）
      · 窗口切成 chunk，完成即落盘（分片文件原子改名），state.json 记录
        已完成 chunk —— 重跑自动跳过
      · 聚合器每 SWEEP_SNAPSHOT_EVERY 个 chunk 存一次快照；晚于快照完成的
        chunk 在恢复时重扫，保证「快照 ⇔ 已完成集合」严格配对，绝不双计
      · 单 chunk 失败不拖垮整趟：并发聚合一律「chunk 内局部聚合 → 完成后
        持锁合并」，末尾对失败 chunk 重试 3 轮，仍失败则报出续跑指引

    ============================================================
    四、 试点标定（2026-09-07，决定了本文件的所有体量参数）
    ============================================================
      · auth 日志 ~4.6MB/6k 块 ⇒ main 趟全窗口 ≈ 6-8 GB，~19k 请求
      · swap 日志 59.4 条/块   ⇒ mev 隔桶后 ≈ 15-20 GB，~6k 请求
      · 无过滤交易流 706 笔/块 ⇒ 全窗口 1.7 TB，「无过滤全量趟」不可行 ——
        这就是 human 必须走「探针 + 地址过滤分批」的原因
      · 请求吞吐 ≈ 5.2 req/s（8 并发），全流程 ≈ 25 万请求 ≈ 13-15 小时纯网络，
        加 SQD 过载余量按 1-2 个通宵计；全程可中断续跑

    ============================================================
    五、 seg 字段的对齐原则（特征正确性的关键）
    ============================================================
      f01_latency 只在同 seg 内算 Δt（跨段是人为断点，不是真实间隔）。
      因此 **seg 必须精确反映「哪里有真实的采集断点」**:
        agent/olas  全窗口连续 → seg = 0（整条流都是真实相邻）
        human       全窗口连续 → seg = 0
        bot         36 个采集段 → seg = 采集段序号（段间是真断点）
      🔴 例外: 触发 3,000 条封顶时，抽稀本身制造了新断点 ——
         此时才把 seg 改成封顶桶号（连续组用 24 等分网格桶），
         既保跨度又不伪造相邻关系。未封顶的地址不受影响。
      三组 win_lo/win_hi 一律 = 完整建模窗口（等长，周期特征可比）。
'''

import argparse
import bisect
import gzip
import json
import random
import shutil
import sys
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402
from src.utils._base import log, topic0, write_csv, read_source_csv  # noqa: E402
from src.utils.common import read_address_csv, save_cache  # noqa: E402
from src.utils.sqd_client import SQDClient  # noqa: E402

USDC = '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913'
T_AUTH = topic0('AuthorizationUsed(address,bytes32)')
T_V2 = topic0('Swap(address,uint256,uint256,uint256,uint256,address)')
T_V3 = topic0('Swap(address,address,int256,int256,uint160,uint128,int24)')
T_V4 = topic0('Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)')
T_EXEC = topic0('ExecutionSuccess(bytes32,uint256)')

TX_FIELDS_FULL = {
    'block': {'number': True, 'timestamp': True},
    'transaction': {'from': True, 'to': True, 'hash': True, 'nonce': True,
                    'gasPrice': True, 'gasUsed': True, 'value': True,
                    'status': True, 'transactionIndex': True, 'sighash': True},
}
# collect 阶段单 chunk 内每地址最多留多少条 —— bot/olas 用。
# 3,000 封顶 / 36 段 ≈ 83，×3 倍余量供装配端去重后精确截断。
CHUNK_ADDR_CAP = 250


# ================================================================
#  小工具
# ================================================================
def t2a(t: str) -> str:
    """32 字节 topic → 20 字节地址（小写）。"""
    return ('0x' + t[-40:]).lower()


def to_dec(v) -> str:
    """SQD 十六进制 → 十进制字符串（与 Etherscan 同口径）。"""
    if v is None:
        return '0'
    if isinstance(v, int):
        return str(v)
    s = str(v)
    try:
        return str(int(s, 16)) if s.startswith('0x') else str(int(s))
    except ValueError:
        return '0'


def safe_int(v, default: int = 0) -> int:
    try:
        if v is None or v == '':
            return default
        return int(v, 16) if isinstance(v, str) and v.startswith('0x') else int(v)
    except (TypeError, ValueError):
        return default


def is_addr(a) -> bool:
    a = str(a or '').lower()
    return a.startswith('0x') and len(a) == 42


def rec_of(kind: str, hdr: dict, tx: dict = None) -> dict:
    """统一活动记录（紧凑键，装配时展开）；字段语义与 fetch_activity 一致。"""
    tx = tx or {}
    return {
        't': safe_int(hdr.get('timestamp')),
        'b': safe_int(hdr.get('number')),
        'x': safe_int(tx.get('transactionIndex')),
        'k': kind,
        'to': (tx.get('to') or '').lower(),
        'v': to_dec(tx.get('value')),
        'g': to_dec(tx.get('gasPrice')),
        'n': safe_int(tx.get('nonce')),
        # SQD status(1=成功) 与 Etherscan isError(0=成功) 语义相反
        'e': '0' if safe_int(tx.get('status'), 1) == 1 else '1',
        'm': tx.get('sighash') or '0x',
        'h': tx.get('hash') or '',
    }


def expand_rec(r: dict) -> dict:
    """紧凑记录 → compute_features 的输入契约字段名。"""
    return {
        'timeStamp': r['t'], 'blockNumber': r['b'], 'txIndex': r['x'],
        'kind': r['k'], 'to': r['to'], 'value': r['v'], 'gasPrice': r['g'],
        'nonce': r['n'], 'isError': r['e'], 'methodId': r['m'],
        'hash': r.get('h', ''),
    }


def chunk_ranges(lo: int, hi: int, size: int = None) -> list:
    """[lo, hi) 均切成 size 块的 (序号, 起, 止) 列表（止为闭区间）。"""
    size = size or config.SWEEP_CHUNK_BLOCKS
    out, cur, i = [], lo, 0
    while cur < hi:
        out.append((i, cur, min(cur + size, hi) - 1))
        cur += size
        i += 1
    return out


def spread_segments(lo: int, hi: int, n_seg: int, seg_blocks: int) -> list:
    """窗口内均布 n_seg 个 seg_blocks 长的连续段（首段贴头、末段贴尾）。"""
    span = hi - lo
    seg_blocks = min(seg_blocks, span)
    if n_seg <= 1 or span <= seg_blocks:
        return [(0, lo, hi - 1)]
    step = (span - seg_blocks) / (n_seg - 1)
    if step < seg_blocks:          # 窗口太小时收窄段长，首尾相接不重叠
        seg_blocks = max(int(step), 1)
    out, seen = [], set()
    for i in range(n_seg):
        s = int(lo + i * step)
        if s not in seen:
            seen.add(s)
            out.append((len(out), s, min(s + seg_blocks, hi) - 1))
    return out


def seg_mapper(segs: list):
    """按段起点二分，把区块号映射到所属采集段序号（seg 对齐原则）。"""
    starts = [s for _, s, _ in segs]

    def f(bn: int) -> int:
        i = bisect.bisect_right(starts, bn) - 1
        return max(0, min(i, len(segs) - 1))
    return f


# ================================================================
#  SweepState — 断点状态 + 分片目录
# ================================================================
class SweepState(object):
    """
    一个阶段一个状态目录: data/sweep/{name}_{lo}_{hi}/
      state.json               {"done": [chunk_id...]}
      agg.json.gz              聚合快照 {"done": [...], "data": {...}}
      shards/{cid}.ndjson.gz   活动记录分片（每行 {a: 地址, ...紧凑记录}）
    """

    def __init__(self, name: str, lo: int, hi: int, force: bool = False):
        self.dir = config.SWEEP_DIR / f'{name}_{lo}_{hi}'
        if force and self.dir.exists():
            shutil.rmtree(self.dir)
        (self.dir / 'shards').mkdir(parents=True, exist_ok=True)
        self.state_p = self.dir / 'state.json'
        self.agg_p = self.dir / 'agg.json.gz'
        self.done = set()
        if self.state_p.exists():
            try:
                self.done = set(json.loads(self.state_p.read_text())['done'])
            except Exception:
                self.done = set()
        self.lock = threading.Lock()

    def ensure_plan(self, ranges: list, extra: dict = None) -> None:
        """
        🔴 计划指纹守卫 —— 分段配置一变，旧分片必须作废。

        【为什么必须有】 分片是按 chunk_id 命名的。若改了段数/段长/配额，
        新计划的 chunk_id 与旧的对不上: 旧分片不会被覆盖，却仍会被
        iter_shards 读进装配 —— 于是同一地址混进两套采样口径的数据，
        seg 语义错乱、Δt 出现凭空的"跨段"。这类脏数据不会报错，
        只会安静地污染特征，是最难查的一类问题。

        同理，聚合结构本身改版（如 payers 从 dict 明细改成计数）时，
        旧快照的字段语义已经不同，也必须作废 —— 所以 extra 里带上格式版本。

        【做法】 把「chunk 数 + 首尾 id + 总块数 + 格式版本」存进 plan.json，
        不一致就清空该阶段的 shards 与 state，从头重扫（其他阶段不受影响）。
        """
        plan = {'n': len(ranges),
                'first': str(ranges[0][0]) if ranges else '',
                'last': str(ranges[-1][0]) if ranges else '',
                'blocks': sum(h - l + 1 for _, l, h in ranges)}
        if extra:
            plan.update(extra)
        pf = self.dir / 'plan.json'
        if pf.exists():
            try:
                old_plan = json.loads(pf.read_text())
            except Exception:
                old_plan = None
            if old_plan != plan:
                log(f'  [{self.dir.name}] 分段配置已变更'
                    f'（{old_plan} → {plan}），旧分片作废并重扫', 'warn')
                shutil.rmtree(self.dir / 'shards', ignore_errors=True)
                (self.dir / 'shards').mkdir(parents=True, exist_ok=True)
                self.done = set()
                for f in (self.state_p, self.agg_p):
                    if f.exists():
                        f.unlink()
        pf.write_text(json.dumps(plan))

    def mark_done(self, cid: str) -> None:
        with self.lock:
            self.done.add(cid)
            tmp = self.state_p.with_suffix('.tmp')
            tmp.write_text(json.dumps({'done': sorted(self.done)}))
            tmp.replace(self.state_p)

    def drop_stale(self, snap_done: set, label: str) -> None:
        """晚于快照完成的 chunk 聚合已丢失 —— 从 done 中剔除，恢复时重扫。"""
        stale = self.done - snap_done
        if stale:
            log(f'[{label}] {len(stale)} 个 chunk 晚于聚合快照完成，'
                f'为保一致性重扫（分片会原子覆盖，不会双计）', 'warn')
            self.done -= stale

    def write_shard(self, cid: str, rows: list) -> None:
        """rows: [(addr, 紧凑记录 dict)]，原子写（重扫覆盖旧分片）。"""
        p = self.dir / 'shards' / f'{cid}.ndjson.gz'
        if not rows:
            if p.exists():
                p.unlink()
            return
        tmp = p.with_suffix('.tmp')
        with gzip.open(tmp, 'wt', encoding='utf-8') as f:
            for addr, r in rows:
                f.write(json.dumps({'a': addr, **r}, separators=(',', ':')) + '\n')
        tmp.replace(p)

    def iter_shards(self):
        for p in sorted((self.dir / 'shards').glob('*.ndjson.gz')):
            with gzip.open(p, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def save_agg(self, data) -> None:
        with self.lock:
            tmp = self.agg_p.with_suffix('.tmp')
            with gzip.open(tmp, 'wt', encoding='utf-8') as f:
                json.dump({'done': sorted(self.done), 'data': data}, f,
                          separators=(',', ':'))
            tmp.replace(self.agg_p)

    def load_agg(self):
        """返回 (与快照配对的 done 集合, 聚合数据)；无快照时 (set(), None)。"""
        if not self.agg_p.exists():
            return set(), None
        try:
            with gzip.open(self.agg_p, 'rt', encoding='utf-8') as f:
                snap = json.load(f)
            return set(snap.get('done', [])), snap.get('data')
        except Exception as e:
            log(f'聚合快照损坏，忽略并重算: {e}', 'warn')
            return set(), None


# ================================================================
#  ChunkedSweep — 并行分块扫描执行器（子类实现 _run_chunk）
# ================================================================
class ChunkedSweep(object):
    """
    把若干 (chunk_id, lo, hi) 范围用 SQD_WORKERS 并发扫完。

    并发纪律（🔴 数据正确性的根基）:
      · chunk 内一律局部聚合，chunk 完成后在 self.agg_lock 下合并到全局
      · 快照 snapshot() 也在 agg_lock 下执行 ⇒ 永远不会 dump 到一半被改
      · 分片原子落盘 + state 标记完成 ⇒ 重跑幂等
    """

    def __init__(self, state: SweepState, label: str):
        self.state = state
        self.label = label
        self.client = SQDClient()
        self.agg_lock = threading.Lock()
        self._t0 = time.time()
        self._n_req = 0
        self._req_lock = threading.Lock()

    # ---- 子类实现 ----
    def _run_chunk(self, cid: str, lo: int, hi: int) -> None:
        raise NotImplementedError

    def snapshot(self) -> None:
        """在 agg_lock 下被调用；子类覆写以持久化聚合。"""

    # ---- 通用行走 ----
    def _walk(self, body: dict, lo: int, hi: int, on_block) -> None:
        """串行续拉 [lo, hi]，每个区块回调 on_block(blk)。"""
        b = dict(body)
        cur = lo
        while cur <= hi:
            b['fromBlock'] = cur
            b['toBlock'] = hi
            raw = self.client._post('/stream', b)
            with self._req_lock:
                self._n_req += 1
            blocks = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
            if not blocks:
                break
            for blk in blocks:
                on_block(blk)
            nxt = blocks[-1]['header']['number'] + 1
            if nxt <= cur:
                break
            cur = nxt

    # 子类覆写以把「聚合结构版本」等纳入计划指纹（变更即作废旧状态）
    PLAN_EXTRA = None

    def run(self, ranges: list) -> bool:
        # 分段配置或聚合格式变更 ⇒ 旧分片与旧快照一并作废
        self.state.ensure_plan(ranges, self.PLAN_EXTRA)
        todo = [(c, l, h) for c, l, h in ranges
                if str(c) not in self.state.done]
        if not todo:
            log(f'[{self.label}] 全部 {len(ranges)} 个 chunk 已完成'
                f'（断点续跑跳过）', 'ok')
            return True
        est_req = sum(h - l for _, l, h in todo) / config.SQD_CHUNK_BLOCKS
        log(f'[{self.label}] 待扫 {len(todo)}/{len(ranges)} 个 chunk，'
            f'约 {est_req:,.0f} 次请求 ≈ {est_req/5.2/60:.0f} 分钟'
            f'（SQD 过载时更久；可随时 Ctrl-C，重跑续采）', 'step')

        failed, since_snap = todo, 0
        for round_i in range(3):
            if not failed:
                break
            if round_i:
                log(f'[{self.label}] 第 {round_i+1} 轮重试 '
                    f'{len(failed)} 个失败 chunk', 'warn')
            cur, failed = failed, []
            done_n = len(ranges) - len(cur)
            ex = ThreadPoolExecutor(max_workers=config.SQD_WORKERS)
            try:
                futs = {ex.submit(self._run_chunk, str(c), l, h): (c, l, h)
                        for c, l, h in cur}
                for fut in as_completed(futs):
                    c, l, h = futs[fut]
                    try:
                        fut.result()
                        self.state.mark_done(str(c))
                        done_n += 1
                        since_snap += 1
                        if since_snap >= config.SWEEP_SNAPSHOT_EVERY:
                            with self.agg_lock:
                                self.snapshot()
                            since_snap = 0
                        if done_n % 20 == 0 or done_n == len(ranges):
                            el = time.time() - self._t0
                            rate = self._n_req / max(el, 1)
                            eta = ((len(ranges) - done_n)
                                   * el / max(done_n, 1))
                            log(f'  [{self.label}] {done_n}/{len(ranges)} chunk'
                                f' | {self._n_req:,} 请求 {rate:.1f} req/s'
                                f' | 剩余约 {eta/60:.0f} 分钟')
                    except Exception as e:
                        failed.append((c, l, h))
                        log(f'  [{self.label}] chunk {c} 失败（稍后重试）: '
                            f'{type(e).__name__}: {str(e)[:80]}', 'warn')
            except KeyboardInterrupt:
                log(f'[{self.label}] 用户中断 —— 已完成的 chunk 均已落盘，'
                    f'重跑同一命令续采', 'warn')
                ex.shutdown(wait=False, cancel_futures=True)
                with self.agg_lock:
                    self.snapshot()
                raise
            ex.shutdown(wait=True)
        with self.agg_lock:
            self.snapshot()
        if failed:
            log(f'[{self.label}] 🔴 {len(failed)} 个 chunk 三轮后仍失败 —— '
                f'网络恢复后重跑同一命令即可续采', 'err')
            return False
        log(f'[{self.label}] 完成（{self._n_req:,} 次请求，'
            f'{(time.time()-self._t0)/60:.0f} 分钟）', 'ok')
        return True


# ================================================================
#  地址池读取
# ================================================================
def olas_base_set() -> set:
    """Olas 注册表里的 Base 链 Safe（排除 gnosis 行）。"""
    out = set()
    for r in read_source_csv('olas_agents.csv'):
        if 'gnosis' not in (r.get('source') or '').lower() \
                and is_addr(r.get('address')):
            out.add(r['address'].lower())
    return out


def hot_wallet_list() -> list:
    return [r['address'].lower() for r in read_source_csv('cex_hot_wallets.csv')
            if is_addr(r.get('address'))]


def facilitator_avgs() -> dict:
    """x402_facilitators.csv → {submitter: avg_usd(float) 或 None}。"""
    out = {}
    for r in read_source_csv('x402_facilitators.csv'):
        a = (r.get('address') or '').lower()
        if not is_addr(a):
            continue
        try:
            out[a] = float(r.get('avg_usd'))
        except (TypeError, ValueError):
            out[a] = None
    return out


# ================================================================
#  阶段① hotwallets — 窗口内均布切片反推 CEX 热钱包
# ================================================================
def phase_hotwallets(lo: int, hi: int, force: bool) -> bool:
    from src.data_fetch.fetch_human import HumanDataSource

    st = SweepState('hotwallets', lo, hi, force)
    _, saved = st.load_agg()
    merged = {k: int(v) for k, v in (saved or {}).items()}

    segs = spread_segments(lo, hi, config.HOT_WALLET_SLICES,
                           config.HOT_WALLET_SLICE_BLOCKS)
    for i, s_lo, s_hi in segs:
        cid = f'slice{i}'
        if cid in st.done:
            continue
        log(f'[hotwallets] 切片 {i+1}/{len(segs)} 区块 {s_lo:,}→{s_hi:,}', 'step')
        ds = HumanDataSource(lo=s_lo, hi=s_hi)
        hot = ds._find_hot_wallets(s_lo, s_hi)
        for a, n in hot.items():
            merged[a] = max(merged.get(a, 0), int(n))
        st.mark_done(cid)
        st.save_agg(merged)

    top = sorted(merged.items(), key=lambda x: -x[1])[:config.MAX_HOT_WALLETS_TOTAL]
    write_csv('cex_hot_wallets.csv',
              [{'address': a, 'source': 'cex_hot_wallet',
                'note': f'recipients={n}', 'n_recipients': n} for a, n in top],
              ['address', 'source', 'note', 'n_recipients'],
              note=('CEX 热钱包（散播型行为反推，窗口内均布切片合并）\n'
                    '判据: 单切片内 ≥40 个不同接收方且唯一接收方占比 ≥0.8\n'
                    f'窗口 {lo}-{hi} | {len(segs)} 个切片'))
    log(f'[hotwallets] 合并得到 {len(top)} 个热钱包', 'ok')
    return len(top) > 0


# ================================================================
#  阶段② main — 全窗口一趟：x402 发现+事件活动 / Olas exec / 提币接收方
# ================================================================
class MainSweep(ChunkedSweep):
    """
    全窗口一趟：x402 付款方注册表 + 事件活动 + Olas 执行事件 + 提币接收方。

    🔴 内存设计（决定这一趟能不能跑完整窗口）:
      实测密度外推，187 天窗口会累积 100 万~200 万个去重付款方。
      若每个付款方都存 {提交者: 次数} 字典，光注册表就要 1 GB 以上，
      且每次快照要序列化整个结构。

      改为**紧凑四元组** [n, first, last, n_bad]:
        n_bad = 经「零售型提交者」结算的笔数 —— 提纯只需要这个比例，
        而零售型名单在 sources 阶段就已产出（x402_facilitators.csv），
        因此可以在采集时就地判定，不必留着原始 {提交者: 次数}。
      每个付款方 4 个整数 ⇒ 200 万个约 300 MB，快照也小得多。

      代价: 失去「每个付款方用了哪些提交者」的明细。
      折中: 只为**已越过有效门槛**的付款方（实测约 3%）保留提交者明细，
      供论文的「运营者聚类」护栏使用。
    """

    def __init__(self, state, hot: list, olas: set, bad_subs: set = None):
        super().__init__(state, 'main')
        self.hot = hot
        self.hot_set = set(hot)
        self.olas = olas
        self.bad_subs = bad_subs or set()
        # 计划指纹的附加项:
        #   agg_fmt  聚合结构版本（v2 = payers 存 [n, first, last, n_bad]）
        #   bad_subs 零售型提交者名单的指纹 —— 🔴 n_bad 是在采集时就地累计的，
        #     名单一变，已存的 n_bad 就不再对应当前判据。若不作废重扫，
        #     提纯会**静默**按旧名单生效（例如用户只删了 facilitators.csv
        #     重跑 sources，main 却因已完成而跳过）。宁可多扫一趟。
        self.PLAN_EXTRA = {
            'agg_fmt': 2,
            'bad_subs': f'{len(self.bad_subs)}:'
                        f'{zlib.crc32(",".join(sorted(self.bad_subs)).encode()):08x}',
        }
        self.min_n = config.MIN_TX_FOR_FEATURES
        # 全局聚合（只在 agg_lock 下修改）
        self.payers = {}      # addr -> [n, first, last, n_bad]
        self.subs_big = {}    # addr -> {submitter: n}（仅 n>=min_n 的付款方）
        self.recips = {}      # addr -> [n, first, last]

    def _query(self, lo, hi):
        logs = [{'address': [USDC], 'topic0': [T_AUTH], 'transaction': True}]
        if config.SAFE_EXEC_TOPIC and self.olas:
            logs.append({'topic0': [T_EXEC]})
        q = {
            'type': 'evm', 'fromBlock': lo, 'toBlock': hi,
            'fields': {
                'block': {'number': True, 'timestamp': True},
                'log': {'address': True, 'topics': True,
                        'transactionIndex': True},
                'transaction': TX_FIELDS_FULL['transaction'],
            },
            'logs': logs,
        }
        if self.hot:
            q['transactions'] = [{'from': self.hot}]
        return q

    def _run_chunk(self, cid, lo, hi):
        l_pay = defaultdict(lambda: [0, 0, 0, defaultdict(int)])
        l_rec = defaultdict(lambda: [0, 0, 0])
        rows = []

        def on_block(blk):
            hdr = blk.get('header', {})
            ts = safe_int(hdr.get('timestamp'))
            txs = blk.get('transactions', []) or []
            txmap = {safe_int(t.get('transactionIndex')): t for t in txs}
            for lg in blk.get('logs', []) or []:
                tp = lg.get('topics') or []
                if not tp:
                    continue
                txi = safe_int(lg.get('transactionIndex'))
                if tp[0] == T_AUTH and len(tp) >= 2:
                    payer = t2a(tp[1])
                    sub = (txmap.get(txi, {}).get('from') or '').lower()
                    p = l_pay[payer]
                    p[0] += 1
                    p[1] = ts if not p[1] else min(p[1], ts)
                    p[2] = max(p[2], ts)
                    if sub:
                        p[3][sub] += 1
                    r = rec_of('event', hdr, {'transactionIndex': txi})
                    r['m'] = tp[0][:10]
                    rows.append((payer, r))
                elif tp[0] == T_EXEC:
                    safe = (lg.get('address') or '').lower()
                    if safe in self.olas:
                        r = rec_of('event', hdr, {'transactionIndex': txi})
                        r['m'] = tp[0][:10]
                        rows.append((safe, r))
            if self.hot_set:
                for t in txs:
                    f = (t.get('from') or '').lower()
                    if f not in self.hot_set:
                        continue
                    to = (t.get('to') or '').lower()
                    if is_addr(to) and to not in self.hot_set:
                        rr = l_rec[to]
                        rr[0] += 1
                        rr[1] = ts if not rr[1] else min(rr[1], ts)
                        rr[2] = max(rr[2], ts)

        self._walk(self._query(lo, hi), lo, hi, on_block)
        self.state.write_shard(cid, rows)
        with self.agg_lock:
            for a, v in l_pay.items():
                g = self.payers.setdefault(a, [0, 0, 0, 0])
                g[0] += v[0]
                g[1] = v[1] if not g[1] else min(g[1], v[1] or g[1])
                g[2] = max(g[2], v[2])
                for sub, n in v[3].items():
                    if sub in self.bad_subs:
                        g[3] += n
                # 只为越过门槛的付款方留提交者明细（供运营者聚类；约 3% 的量）
                if g[0] >= self.min_n:
                    d = self.subs_big.setdefault(a, {})
                    for sub, n in v[3].items():
                        d[sub] = d.get(sub, 0) + n
            for a, v in l_rec.items():
                g = self.recips.setdefault(a, [0, 0, 0])
                g[0] += v[0]
                g[1] = v[1] if not g[1] else min(g[1], v[1] or g[1])
                g[2] = max(g[2], v[2])

    def snapshot(self):
        self.state.save_agg({'payers': self.payers, 'recips': self.recips,
                             'subs_big': self.subs_big})


def phase_main(lo: int, hi: int, force: bool) -> bool:
    st = SweepState('main', lo, hi, force)
    hot = hot_wallet_list()
    olas = olas_base_set()
    if not hot:
        log('[main] 无热钱包清单（先跑 hotwallets）—— 本趟将不采提币接收方',
            'warn')
    log(f'[main] 热钱包 {len(hot)} 个 | Olas Base Safe {len(olas)} 个', 'info')

    # 零售型提交者名单在 sources 阶段已产出，采集时就地判定「被污染的支付」，
    # 这样注册表只需存计数而不必留 {提交者: 次数} 明细（见 MainSweep 文档）。
    bad_subs = {a for a, avg in facilitator_avgs().items()
                if avg is not None and avg >= config.X402_FACIL_BAD_MIN_AVG}
    if not bad_subs:
        log('[main] ⚠️ 没有零售型提交者名单（x402_facilitators.csv 缺失或无 '
            'avg_usd 列）—— agent 组将无法按提交者画像提纯。'
            '建议先跑: python3 src/data_fetch/fetch_x402.py --profile', 'warn')
    else:
        log(f'[main] 零售型提交者 {len(bad_subs)} 个（其结算的支付将计入 n_bad）',
            'info')

    # 🔴 顺序要紧: 必须先校验计划（分段配置/聚合格式变了就作废旧状态），
    #   再读快照。反过来会把已作废的旧格式数据先读进内存，然后在合并时炸掉。
    ranges = chunk_ranges(lo, hi)
    sweep = MainSweep(st, hot, olas, bad_subs)
    st.ensure_plan(ranges, sweep.PLAN_EXTRA)
    snap_done, saved = st.load_agg()
    # 🔴 除了计划指纹，还要校验**数据本身**的结构。
    #   计划指纹只在「计划变了」时作废状态；但如果一次运行在计划已更新之后
    #   把内存里的旧结构又写回快照，指纹就检不出来了。
    #   这里直接抽查一条 payers 记录: v[3] 必须是数字(n_bad)，不是旧的字典。
    if saved:
        probe_v = next(iter((saved.get('payers') or {}).values()), None)
        if probe_v is not None and not isinstance(probe_v[3], (int, float)):
            log('[main] 快照是旧结构（payers[3] 不是 n_bad 计数），'
                '丢弃并重扫本趟', 'warn')
            saved, snap_done = None, set()
            st.done = set()
    if saved:
        sweep.payers = {a: list(v) for a, v in saved.get('payers', {}).items()}
        sweep.subs_big = {a: dict(v)
                          for a, v in (saved.get('subs_big') or {}).items()}
        sweep.recips = {a: list(v) for a, v in saved.get('recips', {}).items()}
        log(f'[main] 从快照恢复: 付款方 {len(sweep.payers):,} / '
            f'接收方 {len(sweep.recips):,}', 'info')
    st.drop_stale(snap_done, 'main')

    if not sweep.run(ranges):
        return False

    # ---- 产出 1: x402 付款方清单（只写 ≥ 有效门槛的，其余永远选不上）----
    min_n = config.X402_MIN_PAYMENTS_CSV
    rows = []
    for a, (n, first, last, n_bad) in sweep.payers.items():
        if n < min_n:
            continue
        subs = sweep.subs_big.get(a) or {}
        top_sub = max(subs, key=subs.get) if subs else ''
        rows.append({'address': a, 'source': 'x402_payer',
                     'note': f'n={n} span={(last-first)/86400:.0f}d '
                             f'bad={n_bad} submitters={len(subs)}',
                     'n_payments': n, 'avg_usd': -1, 'n_sellers': -1,
                     'bad_share': round(n_bad / max(n, 1), 4),
                     'top_submitter': top_sub})
    rows.sort(key=lambda r: -r['n_payments'])
    write_csv('x402_payers.csv', rows,
              ['address', 'source', 'note', 'n_payments', 'avg_usd',
               'n_sellers', 'bad_share', 'top_submitter'],
              note=('x402 付款方（agent 弱正样本，真值质量 ★★）\n'
                    '★ 全窗口 AuthorizationUsed 扫描（sweep_window main 趟）\n'
                    f'  去重付款方 {len(sweep.payers):,} 个，其中窗口内 ≥{min_n} '
                    f'次的 {len(rows):,} 个（低于门槛的不写入）\n'
                    'avg_usd=-1: 金额提纯改用提交者画像'
                    '（fetch_x402 --profile，见 x402_facilitators.csv）\n'
                    f'区块 {lo}-{hi}'))

    # ---- 产出 2: human 候选池 ----
    # 🔴 免费预排序: 按「收到多少次 CEX 提币」降序取前 N。
    #   收提币越频繁 ⇒ 越可能是在用的活钱包 ⇒ 越可能过 ≥20 活动门槛。
    #   这一步不额外花任何请求（main 趟已顺带统计），却能把后续昂贵的
    #   全窗口采集的候选量压掉一个量级。
    #   ⚠️ 它只用「次数」这一个标量，不涉及任何时间分布 —— 反循环设计不变。
    rng = random.Random(config.RANDOM_SEED)
    cand = list(sweep.recips.items())
    if getattr(config, 'HUMAN_PRERANK_BY_WITHDRAWALS', True):
        cand.sort(key=lambda kv: (-kv[1][0], rng.random()))
        rank_note = '按收到提币次数降序预排序'
    else:
        rng.shuffle(cand)
        rank_note = '种子随机采样'
    cand = cand[:config.HUMAN_PROBE_POOL]
    write_csv('human_candidates.csv',
              [{'address': a, 'source': 'base_cex_withdrawal',
                'note': f'wd={v[0]}', 'n_from_cex': v[0]} for a, v in cand],
              ['address', 'source', 'note', 'n_from_cex'],
              note=('human 候选池（热钱包提币接收方，待探针筛活跃度）\n'
                    f'全部接收方 {len(sweep.recips):,} 个，'
                    f'{rank_note}取 {len(cand):,} 个进探针\n'
                    '🔴 只按活动量筛，不看时间分布（反循环设计）\n'
                    f'区块 {lo}-{hi} | seed={config.RANDOM_SEED}'))
    log(f'[main] 付款方 {len(sweep.payers):,}（≥{min_n} 次: {len(rows):,}）| '
        f'提币接收方 {len(sweep.recips):,}（入探针 {len(cand):,}）', 'ok')
    return True


# ================================================================
#  阶段③ mev — 隔桶抽样识别原子套利 bot
# ================================================================
class MevSweep(ChunkedSweep):
    def __init__(self, state, stride):
        super().__init__(state, f'mev(桶{config.BOT_BUCKET_BLOCKS}块'
                                f'×stride{stride})')
        self.stats = {}     # sender -> [total_arb, best_bucket, n_blocks, max_sw]

    def _run_chunk(self, cid, lo, hi):
        local = defaultdict(lambda: [0, set(), 0])   # arb, blocks, max_swaps

        def on_block(blk):
            hdr = blk.get('header', {})
            bn = safe_int(hdr.get('number'))
            txmap = {safe_int(t.get('transactionIndex')):
                     (t.get('from') or '').lower()
                     for t in blk.get('transactions', []) or []}
            per_tx = defaultdict(int)
            for lg in blk.get('logs', []) or []:
                per_tx[safe_int(lg.get('transactionIndex'))] += 1
            for txi, cnt in per_tx.items():
                if cnt < config.BOT_MIN_SWAPS_IN_TX:
                    continue
                sender = txmap.get(txi)
                if not sender:
                    continue
                s = local[sender]
                s[0] += 1
                s[1].add(bn)
                s[2] = max(s[2], cnt)

        q = {
            'type': 'evm', 'fromBlock': lo, 'toBlock': hi,
            'fields': {
                'block': {'number': True, 'timestamp': True},
                'log': {'transactionIndex': True},
                'transaction': {'from': True, 'transactionIndex': True},
            },
            'logs': [{'topic0': [T_V2, T_V3, T_V4], 'transaction': True}],
        }
        self._walk(q, lo, hi, on_block)
        self.state.write_shard(cid, [])
        with self.agg_lock:
            for sender, s in local.items():
                if s[0] < 2:     # 单桶不足 2 笔的永远够不着判据，省内存
                    continue
                g = self.stats.setdefault(sender, [0, 0, 0, 0])
                g[0] += s[0]
                g[1] = max(g[1], s[0])
                g[2] += len(s[1])
                g[3] = max(g[3], s[2])

    def snapshot(self):
        self.state.save_agg(self.stats)


def phase_mev(lo: int, hi: int, force: bool) -> bool:
    st = SweepState('mev', lo, hi, force)
    stride = config.BOT_BUCKET_STRIDE
    bsz = config.BOT_BUCKET_BLOCKS

    ranges, i, cur = [], 0, lo
    while cur < hi:
        ranges.append((f'b{i:05d}', cur, min(cur + bsz, hi) - 1))
        cur += bsz * stride
        i += 1

    sweep = MevSweep(st, stride)
    st.ensure_plan(ranges, MevSweep.PLAN_EXTRA)   # 先校验计划，再读快照
    snap_done, saved = st.load_agg()
    if saved:
        sweep.stats = {a: list(v) for a, v in saved.items()}
        log(f'[mev] 从快照恢复: 候选 {len(sweep.stats):,}', 'info')
    st.drop_stale(snap_done, 'mev')

    if not sweep.run(ranges):
        return False

    rows = []
    for a, (total, best, n_blocks, max_sw) in sweep.stats.items():
        if best < config.BOT_MIN_BUCKET_ARB or n_blocks < config.BOT_MIN_BLOCKS:
            continue
        rows.append({'address': a, 'source': 'mev_atomic_arb',
                     'note': f'arb_tx={total} max_swaps={max_sw} '
                             f'blocks={n_blocks} best_bucket={best}',
                     'arb_tx': total, 'max_swaps': max_sw,
                     'n_blocks': n_blocks})
    rows.sort(key=lambda r: -r['arb_tx'])
    write_csv('mev_bots.csv', rows,
              ['address', 'source', 'note', 'arb_tx', 'max_swaps', 'n_blocks'],
              note=('MEV bot（传统脚本 bot 负样本，真值质量 ★★★）\n'
                    '判据: 单笔交易内 ≥2 次 DEX Swap，且单个 5,000 块桶内 ≥5 笔\n'
                    '  —— 与原 5,000 块窗口判据同紧度（强度等价缩放），\n'
                    '  避免 187 天窗口把偶发多跳 swap 的普通用户收进来\n'
                    f'隔桶抽样 stride={stride}（覆盖 1/{stride} 窗口）\n'
                    f'区块 {lo}-{hi} | 候选 {len(sweep.stats):,} '
                    f'→ 达标 {len(rows):,}'))
    log(f'[mev] 候选 {len(sweep.stats):,} → 达标 bot {len(rows):,}', 'ok')
    return len(rows) > 0


# ================================================================
#  阶段④ probe — human 候选活跃度探针
# ================================================================
class ProbeSweep(ChunkedSweep):
    def __init__(self, state, batches, batch_of):
        super().__init__(state, f'probe({len(batches)}批)')
        self.batches = batches
        self.batch_of = batch_of
        self.hits = {}

    def _run_chunk(self, cid, lo, hi):
        batch = self.batches[self.batch_of[cid]]
        bset = set(batch)
        local = defaultdict(int)

        def on_block(blk):
            for t in blk.get('transactions', []) or []:
                f = (t.get('from') or '').lower()
                if f in bset:
                    local[f] += 1

        q = {
            'type': 'evm', 'fromBlock': lo, 'toBlock': hi,
            'fields': {'block': {'number': True},
                       'transaction': {'from': True}},
            'transactions': [{'from': batch}],
        }
        self._walk(q, lo, hi, on_block)
        with self.agg_lock:
            for a, n in local.items():
                self.hits[a] = self.hits.get(a, 0) + n

    def snapshot(self):
        self.state.save_agg(self.hits)


def phase_probe(lo: int, hi: int, force: bool, keep: int = None) -> bool:
    st = SweepState('probe', lo, hi, force)
    keep = keep or config.HUMAN_PROBE_KEEP
    cand = [r for r in read_source_csv('human_candidates.csv')
            if is_addr(r.get('address'))]
    if not cand:
        log('[probe] 无候选池（先跑 main）', 'err')
        return False

    # 排除已带其他标签的地址（含全量付款方注册表，防 agent 混入 human）
    excl = set(hot_wallet_list())
    for fn in ('x402_payers.csv', 'mev_bots.csv', 'olas_agents.csv',
               'x402_facilitators.csv'):
        for r in read_source_csv(fn):
            a = (r.get('address') or '').lower()
            if is_addr(a):
                excl.add(a)
    _, main_agg = SweepState('main', lo, hi).load_agg()
    for a, v in ((main_agg or {}).get('payers') or {}).items():
        if v[0] >= 5:
            excl.add(a)
    addrs = [r['address'].lower() for r in cand
             if r['address'].lower() not in excl]
    wd = {r['address'].lower(): r.get('n_from_cex', '') for r in cand}
    log(f'[probe] 候选 {len(cand):,} → 排除他组标签后 {len(addrs):,}', 'info')

    segs = spread_segments(lo, hi, config.HUMAN_PROBE_SEGMENTS,
                           config.HUMAN_PROBE_SEG_BLOCKS)
    batches = [addrs[i:i + config.SQD_MAX_ADDR_PER_QUERY]
               for i in range(0, len(addrs), config.SQD_MAX_ADDR_PER_QUERY)]
    ranges, batch_of = [], {}
    for bi in range(len(batches)):
        for si, s_lo, s_hi in segs:
            cid = f'p{bi:03d}s{si:02d}'
            ranges.append((cid, s_lo, s_hi))
            batch_of[cid] = bi

    sweep = ProbeSweep(st, batches, batch_of)
    st.ensure_plan(ranges, ProbeSweep.PLAN_EXTRA)  # 先校验计划，再读快照
    snap_done, saved = st.load_agg()
    if saved:
        sweep.hits = {a: int(v) for a, v in saved.items()}
    st.drop_stale(snap_done, 'probe')

    if not sweep.run(ranges):
        return False

    rng = random.Random(config.RANDOM_SEED)
    ranked = sorted(sweep.hits.items(), key=lambda x: (-x[1], rng.random()))
    top = ranked[:keep]
    p = config.ADDR_DIR / config.GROUPS['human'][0]
    with p.open('w', newline='', encoding='utf-8') as f:
        f.write('# 人类样本候选 —— sweep_window 探针排名（全窗口口径）\n')
        f.write(f'# 探针: {len(segs)} 段 × {config.HUMAN_PROBE_SEG_BLOCKS} 块，'
                f'按命中数排名取前 {keep}\n')
        f.write('# 判据: 收到散播型热钱包转账 + 探针段内有自发交易 + 非他组标签\n')
        f.write('# 🔴 反循环: 只看活动量，绝不看时间分布'
                '（entropy 特征保持独立证据）\n')
        cov = float(config.HUMAN_COVERAGE)
        need = int(config.MIN_TX_FOR_FEATURES / max(cov, 1e-9))
        f.write(f'# 🔴 正式采集覆盖窗口的 {cov:.0%} ⇒ 入选者约需全窗口 ≥{need} 笔，'
                f'样本构成偏向常用钱包（论文需交代）\n')
        f.write('address,source,note\n')
        for a, h in top:
            f.write(f'{a},base_cex_withdrawal,hits={h};wd={wd.get(a, "")}\n')
    log(f'[probe] 有命中 {len(sweep.hits):,} 个；写入前 {len(top):,} '
        f'→ {p.name}', 'ok')
    return len(top) > 0


# ================================================================
#  阶段⑤ collect — 地址过滤分批采集（bot / human / olas）
# ================================================================
class CollectSweep(ChunkedSweep):
    def __init__(self, state, label, batches, batch_of, kind_map, want_to,
                 per_chunk_cap):
        super().__init__(state, label)
        self.batches = batches
        self.batch_of = batch_of
        self.want_to = want_to
        self.cap = per_chunk_cap

    def _run_chunk(self, cid, lo, hi):
        batch = self.batches[self.batch_of[cid]]
        bset = set(batch)
        rows, cnt = [], defaultdict(int)

        def on_block(blk):
            hdr = blk.get('header', {})
            for t in blk.get('transactions', []) or []:
                f = (t.get('from') or '').lower()
                to = (t.get('to') or '').lower()
                if f in bset and (not self.cap or cnt[f] < self.cap):
                    rows.append((f, rec_of('tx_from', hdr, t)))
                    cnt[f] += 1
                if self.want_to and to in bset and to != f \
                        and (not self.cap or cnt[to] < self.cap):
                    rows.append((to, rec_of('tx_to', hdr, t)))
                    cnt[to] += 1

        sel = [{'from': batch}]
        if self.want_to:
            sel.append({'to': batch})
        q = {'type': 'evm', 'fromBlock': lo, 'toBlock': hi,
             'fields': TX_FIELDS_FULL, 'transactions': sel}
        self._walk(q, lo, hi, on_block)
        self.state.write_shard(cid, rows)


def _collect_generic(name: str, lo: int, hi: int, force: bool, addrs: list,
                     segs: list, want_to: bool, per_chunk_cap: int) -> bool:
    st = SweepState(name, lo, hi, force)
    batches = [addrs[i:i + config.SQD_MAX_ADDR_PER_QUERY]
               for i in range(0, len(addrs), config.SQD_MAX_ADDR_PER_QUERY)]
    ranges, batch_of = [], {}
    for bi in range(len(batches)):
        for si, s_lo, s_hi in segs:
            for cj, c_lo, c_hi in chunk_ranges(s_lo, s_hi + 1):
                cid = f'b{bi:03d}s{si:02d}k{cj:02d}'
                ranges.append((cid, c_lo, c_hi))
                batch_of[cid] = bi
    log(f'[{name}] {len(addrs):,} 个地址 / {len(batches)} 批 × {len(segs)} 段',
        'info')
    sweep = CollectSweep(st, name, batches, batch_of, None, want_to,
                         per_chunk_cap)
    return sweep.run(ranges)


def bot_segments(lo, hi):
    return spread_segments(lo, hi, config.BOT_COLLECT_SEGMENTS,
                           config.BOT_COLLECT_SEG_BLOCKS)


def human_segments(lo, hi):
    """
    human 采集段 —— 🔴 覆盖率 1.0 时是「整窗口一段」，这是刻意的。

    人类天然低频（全窗口约 40 笔 ≈ 0.21 笔/天）。若切成多段，
    段内几乎凑不出相邻活动对，f01 的段内 Δt 样本 <5 ⇒ 整组被剔除。
    连续采集时 seg 恒为 0，N 笔活动给出 N-1 个真实 Δt。
    详见 config.HUMAN_COVERAGE 注释。
    """
    cov = float(config.HUMAN_COVERAGE)
    if cov >= 1.0:
        return [(0, lo, hi - 1)]
    n_seg = config.SWEEP_N_SEGMENTS
    seg_len = max(int((hi - lo) * cov / n_seg), 1)
    return spread_segments(lo, hi, n_seg, seg_len)


def phase_collect(lo: int, hi: int, force: bool, quota: dict = None) -> bool:
    quota = quota or config.GROUP_SELECT_QUOTA
    rng = random.Random(config.RANDOM_SEED)
    ok = True

    # ---- bot: 配额 ×1.4 预选（按套利强度四分位轮转，保证强弱都有代表）----
    bots = read_address_csv(config.ADDR_DIR / config.GROUPS['bot'][0])
    if bots:
        def arb_of(r):
            try:
                return int(str(r.get('note', '')).split('arb_tx=')[1].split()[0])
            except (IndexError, ValueError):
                return 0
        bots.sort(key=arb_of)
        n_sel = min(len(bots), int(quota['bot'] * 1.4))
        quart = [bots[i::4] for i in range(4)]
        for qq in quart:
            rng.shuffle(qq)
        sel, qi = [], 0
        while len(sel) < n_sel and any(quart):
            if quart[qi % 4]:
                sel.append(quart[qi % 4].pop())
            qi += 1
        ok &= _collect_generic('collect_bot', lo, hi, force,
                               [r['address'] for r in sel],
                               bot_segments(lo, hi), want_to=True,
                               per_chunk_cap=CHUNK_ADDR_CAP)
    else:
        log('[collect] bot 清单为空（先跑 mev + addresses 合并）', 'err')
        ok = False

    # ---- human: 探针排名全员，全窗口连续采集（见 config.HUMAN_COVERAGE）----
    humans = [r['address'] for r in
              read_address_csv(config.ADDR_DIR / config.GROUPS['human'][0])]
    if humans:
        ok &= _collect_generic('collect_human', lo, hi, force, humans,
                               human_segments(lo, hi), want_to=False,
                               per_chunk_cap=0)
    else:
        log('[collect] human 清单为空（先跑 probe）', 'err')
        ok = False

    # ---- olas/erc8004: 全窗口（★★★ 层，数量小，值得单独一趟）----
    if config.OLAS_COLLECT_FULL_WINDOW:
        agent_rows = read_address_csv(config.ADDR_DIR / config.GROUPS['agent'][0])
        pool = list(dict.fromkeys(
            [r['address'] for r in agent_rows
             if ('olas' in (r.get('source') or '')
                 and 'gnosis' not in (r.get('source') or ''))
             or 'erc8004' in (r.get('source') or '')]))
        if pool:
            ok &= _collect_generic('collect_olas', lo, hi, force, pool,
                                   [(0, lo, hi - 1)], want_to=True,
                                   per_chunk_cap=CHUNK_ADDR_CAP)
    return ok


# ================================================================
#  阶段⑥ assemble — 配额选样 + 落盘 + 漏斗报告
# ================================================================
def phase_assemble(lo: int, hi: int, quota: dict = None) -> bool:
    chain = config.PRIMARY_CHAIN
    quota = dict(quota or config.GROUP_SELECT_QUOTA)
    rng = random.Random(config.RANDOM_SEED)
    min_n = config.MIN_TX_FOR_FEATURES
    t0 = time.time()

    sources = {
        'agent_event': SweepState('main', lo, hi),
        'bot': SweepState('collect_bot', lo, hi),
        'human': SweepState('collect_human', lo, hi),
        'olas': SweepState('collect_olas', lo, hi),
    }

    # ---- 1) x402 付款方注册表 + 提交者提纯 ----
    _, main_agg = sources['agent_event'].load_agg()
    payers_reg = (main_agg or {}).get('payers', {})
    # n_bad 已在 main 趟就地累计（见 MainSweep 文档），这里直接用比例判定。
    # 🔴 旧格式快照（v[3] 是 {提交者:次数} 字典）语义不同，必须挡住 ——
    #   否则提纯会静默失效，agent 组混进零售转账用户却毫无提示。
    _bad_fmt = next((a for a, v in payers_reg.items()
                     if not isinstance(v[3], (int, float))), None)
    if _bad_fmt is not None:
        log('[assemble] 🔴 main 趟的聚合快照是旧格式（v1），提纯无法进行。'
            '请重跑: python3 src/data_fetch/sweep_window.py main --force', 'err')
        return False

    def payer_ok(a):
        """窗口内支付数达标，且经「零售型提交者」结算的占比不超阈值。"""
        v = payers_reg.get(a)
        if not v or v[0] < min_n:
            return False
        return v[3] / max(v[0], 1) <= config.X402_BAD_SUBMITTER_MAX_SHARE

    # ---- 2) 地址清单与来源（agent 清单已由 fetch_all 做过 agent∩bot 冲突剔除）----
    agent_rows = read_address_csv(config.ADDR_DIR / config.GROUPS['agent'][0])
    bot_rows = read_address_csv(config.ADDR_DIR / config.GROUPS['bot'][0])
    human_rows = read_address_csv(config.ADDR_DIR / config.GROUPS['human'][0])
    known = set(config.SQD_DATASETS) | set(config.CHAINS)
    others = {c for c in known if c != chain}
    src_of = {}
    for r in agent_rows:
        src = (r.get('source') or 'x402_payer')
        if any(c in src.lower() for c in others):    # 跨链过滤（gnosis 等）
            continue
        src_of[r['address']] = src
    bot_set = {r['address'] for r in bot_rows}
    human_set = {r['address'] for r in human_rows}
    hot_set = set(hot_wallet_list())
    facil_set = set(facilitator_avgs())

    # ---- 3) 流式重分区（哈希分区），并统计每地址各来源记录数 ----
    part_dir = config.SWEEP_DIR / f'parts_{lo}_{hi}'
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True)
    # 分区数决定装配阶段的内存峰值: 每个分区要整批读进内存做去重与排序。
    # 🔴 256 而非 64 —— 重度 x402 付款方在 187 天里可能有数万条事件，
    #   64 分区时单分区可能驻留几百万条记录（GB 级）。256 分区把每个分区
    #   压到约 120 个地址，峰值稳定在百 MB 量级。（ulimit -n 通常足够）
    n_parts = 256
    writers = [open(part_dir / f'p{i:03d}.ndjson', 'w', encoding='utf-8')
               for i in range(n_parts)]
    counts = {k: defaultdict(int) for k in sources}
    n_lines = 0
    for bucket, stt in sources.items():
        for obj in stt.iter_shards():
            a = obj['a']
            writers[zlib.crc32(a.encode()) % n_parts].write(
                json.dumps(obj, separators=(',', ':')) + '\n')
            counts[bucket][a] += 1
            n_lines += 1
    for w in writers:
        w.close()
    log(f'[assemble] 重分区完成: {n_lines:,} 条记录'
        f'（{(time.time()-t0)/60:.1f} 分钟）', 'ok')

    c_ev, c_bot = counts['agent_event'], counts['bot']
    c_hum, c_olas = counts['human'], counts['olas']

    # ---- 4) 最终选样（先按未去重计数粗筛，写盘时按真实去重数复核）----
    olas_all = {r['address'] for r in agent_rows
                if ('olas' in (r.get('source') or '')
                    and 'gnosis' not in (r.get('source') or ''))
                or 'erc8004' in (r.get('source') or '')}
    olas_eff = [a for a in olas_all
                if c_olas.get(a, 0) + c_ev.get(a, 0) >= min_n]

    x402_pool = [a for a, s in src_of.items()
                 if s == 'x402_payer' and a not in olas_all
                 and a not in bot_set and a not in hot_set
                 and a not in facil_set and payer_ok(a)]
    rng.shuffle(x402_pool)

    bot_pool = [a for a, n in c_bot.items() if n >= min_n and a in bot_set]
    rng.shuffle(bot_pool)

    human_pool = [a for a, n in c_hum.items()
                  if min_n <= n <= config.HUMAN_MAX_ACTS_WINDOW
                  and a in human_set]
    rng.shuffle(human_pool)

    # 配额分配 + 缺口再平衡:
    #   任何一组池子不足配额时，缺口按 QUOTA_REFILL_ORDER（成本从低到高）
    #   补给还有余量的组，保证总量仍然打到 30,000。
    cap_of = {'agent': len(olas_eff) + len(x402_pool),
              'bot': len(bot_pool), 'human': len(human_pool)}
    take = {g: min(quota[g], cap_of[g]) for g in cap_of}
    short = sum(quota.values()) - sum(take.values())
    if short > 0 and config.QUOTA_REBALANCE:
        thin = [g for g in cap_of if take[g] < quota[g]]
        for g in getattr(config, 'QUOTA_REFILL_ORDER', ['agent', 'bot', 'human']):
            if short <= 0:
                break
            room = cap_of[g] - take[g]
            if room <= 0:
                continue
            add = min(short, room)
            take[g] += add
            short -= add
            log(f'[assemble] {"/".join(thin)} 池不足配额，{g} +{add:,} 补足总量',
                'warn')
        if short > 0:
            log(f'[assemble] 三组池子合计仍缺 {short:,} —— '
                f'见末尾的扩池建议', 'warn')

    # 🔴 过选缓冲。上面的池子是按「未去重计数」粗筛的，而写盘时要按
    #   「真实去重数」复核（见第 5 步的 len(uniq) < min_n 分支），两个口径
    #   不一致 ⇒ 少量地址过了粗筛却卡在写盘，最终有效样本少于目标。
    #   实测 30,000 目标下 agent 掉 4 个（29,996）。
    #   对策：按比例过选，写盘时按 target_written 精确截断 ——
    #   无论损耗多少，最终有效样本恰好等于目标。
    target_written = dict(take)
    over = getattr(config, 'SELECT_OVERSAMPLE_RATIO', 0.02)
    take = {g: min(cap_of[g], int(take[g] * (1 + over)) + 20) for g in take}
    log('[assemble] 过选缓冲 +{:.0%}: {} ⇒ 写盘按目标 {} 精确截断'.format(
        over, {g: take[g] for g in take}, target_written), 'info')

    sel_agent = list(olas_eff) + x402_pool[:max(take['agent'] - len(olas_eff), 0)]
    sel_bot = bot_pool[:take['bot']]
    sel_human = human_pool[:take['human']]

    # 🔴 标签冲突守卫。正常情况下不可能发生（fetch_all 已剔除 agent∩bot，
    #   x402_pool 排除了 bot_set/hot_set/facil_set，probe 排除了付款方），
    #   但一旦发生就会「静默」少写几个样本、让漏斗报告对不上账 ——
    #   所以显式检出并报出来，而不是靠 setdefault 悄悄吞掉。
    conflict = ((set(sel_agent) & set(sel_bot))
                | (set(sel_agent) & set(sel_human))
                | (set(sel_bot) & set(sel_human)))
    if conflict:
        log(f'[assemble] 🔴 {len(conflict)} 个地址同时命中多组判据，'
            f'已从所有组剔除（需人工裁决）', 'err')
        sel_agent = [a for a in sel_agent if a not in conflict]
        sel_bot = [a for a in sel_bot if a not in conflict]
        sel_human = [a for a in sel_human if a not in conflict]

    selected = {}
    for a in sel_agent:
        selected[a] = 'agent'
    for a in sel_bot:
        selected[a] = 'bot'
    for a in sel_human:
        selected[a] = 'human'

    log(f'[assemble] 选样: agent {len(sel_agent):,}'
        f'（olas/erc8004 {len(olas_eff)} + x402 {len(sel_agent)-len(olas_eff):,}）'
        f' | bot {len(sel_bot):,}（池 {len(bot_pool):,}）'
        f' | human {len(sel_human):,}（池 {len(human_pool):,}）', 'step')

    # ---- 5) 写 data/raw/{chain}/{group}/*.json（seg 与各组采集段对齐）----
    for g in config.GROUPS:
        d = config.DATA_RAW / chain / g
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # 🔴 seg = 「真实采集断点」的标记，不是「等分网格」——
    #   f01 只在同 seg 内算 Δt，多标一个断点就凭空丢掉一个真实相邻对。
    #   连续采集的组（agent 事件全窗口 / human 全窗口）整条流都是真相邻 ⇒ seg 恒 0；
    #   bot 按 36 段采 ⇒ seg = 段号（段间确实是断点）。
    grid24 = spread_segments(lo, hi, config.SWEEP_N_SEGMENTS,
                             (hi - lo) // config.SWEEP_N_SEGMENTS + 1)
    collect_segs = {'agent': [(0, lo, hi - 1)],
                    'bot': bot_segments(lo, hi),
                    'human': human_segments(lo, hi)}
    mappers = {g: seg_mapper(v) for g, v in collect_segs.items()}
    n_segs_of = {g: len(v) for g, v in collect_segs.items()}
    # 封顶（抽稀）时才需要制造断点：连续组退化成 24 等分网格桶，保跨度
    cap_mappers = {g: (seg_mapper(grid24) if len(v) == 1 else mappers[g])
                   for g, v in collect_segs.items()}
    cap_n_segs = {g: (len(grid24) if len(v) == 1 else len(v))
                  for g, v in collect_segs.items()}
    for g, v in collect_segs.items():
        log(f'  [assemble] {g} 采集段数={len(v)} ⇒ '
            f'{"连续流（seg≡0，全部相邻对有效）" if len(v) == 1 else "分段（段间不计 Δt）"}')

    written = defaultdict(int)
    dropped_thin = defaultdict(int)
    n_over_trim = defaultdict(int)       # 过选余量的丢弃计数（非异常）
    n_capped = 0
    for pi in range(n_parts):
        by_addr = defaultdict(list)
        with open(part_dir / f'p{pi:03d}.ndjson', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                g = selected.get(obj['a'])
                if g is not None:
                    by_addr[(obj['a'], g)].append(obj)
        for (a, g), recs in by_addr.items():
            # 已写满目标 ⇒ 过选出来的余量直接丢弃（见上面的过选缓冲说明）。
            # 分区按地址 crc32 散列，此处截断等价于随机截断，不引入偏置。
            if written[g] >= target_written.get(g, 1 << 30):
                n_over_trim[g] += 1
                continue
            seen, uniq = set(), []
            for r in recs:
                k = (r['t'], r['b'], r['x'], r['k'])
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(r)
            if len(uniq) < min_n:
                dropped_thin[g] += 1
                continue
            uniq.sort(key=lambda r: (r['t'], r['x']))
            cap = config.MAX_ACTIVITY_PER_ADDRESS
            # 未超封顶: seg = 采集段号（连续组恒 0，相邻对全部有效）
            # 超封顶:   抽稀制造了新断点 ⇒ 改用封顶桶号，既保跨度又不伪造相邻
            capped = len(uniq) > cap
            mp = cap_mappers[g] if capped else mappers[g]
            out = []
            for r in uniq:
                e = expand_rec(r)
                e['seg'] = mp(r['b'])
                e['win_lo'], e['win_hi'] = lo, hi
                out.append(e)
            if capped:
                per_seg = max(cap // cap_n_segs[g], 1)
                bucket, kept = defaultdict(int), []
                for e in out:
                    if bucket[e['seg']] < per_seg:
                        bucket[e['seg']] += 1
                        kept.append(e)
                out = kept
                n_capped += 1
            save_cache(g, a, chain, out)
            written[g] += 1
        if (pi + 1) % 32 == 0:
            log('  [assemble] 分区 {}/{} | 已写 '.format(pi + 1, n_parts)
                + ', '.join(f'{g}={n:,}' for g, n in sorted(written.items())))
    shutil.rmtree(part_dir, ignore_errors=True)

    # ---- 6) 漏斗报告（论文表 3.1/3.2 的数据来源）----
    span = hi - lo
    entered = {'agent': len(sel_agent), 'bot': len(sel_bot),
               'human': len(sel_human)}
    got = {'agent': len({a for a in sel_agent
                         if c_ev.get(a, 0) + c_olas.get(a, 0) > 0}),
           'bot': len({a for a in sel_bot if c_bot.get(a, 0) > 0}),
           'human': len({a for a in sel_human if c_hum.get(a, 0) > 0})}
    total_eff = sum(written.values())
    lines = [
        '# 样本漏斗报告（sweep_window 自动生成）', '',
        f'- 建模窗口: 区块 {lo:,} → {hi:,}'
        f'（{span:,} 块 ≈ {span*2/86400:.0f} 天）',
        f'- 有效门槛: 每地址去重活动 ≥ {min_n} 条（前置成入选条件，入选即有效）',
        f'- 生成时间: {time.strftime("%Y-%m-%d %H:%M")}', '',
        '## 表A 发现层（对应论文表 3.1）', '',
        '| 组 | 来源 | 判据 | 池子规模 |',
        '|---|---|---|---|',
        f'| agent | x402 付款方 | 全窗口 AuthorizationUsed，窗口内 ≥{min_n} 次，'
        f'零售型提交占比 ≤{config.X402_BAD_SUBMITTER_MAX_SHARE:.0%} | '
        f'{len(x402_pool):,}（注册表 {len(payers_reg):,}）|',
        f'| agent | Olas/ERC-8004（Base） | 注册表 + 窗口内活动 ≥{min_n} | '
        f'{len(olas_eff):,}/{len(olas_all):,} |',
        f'| bot | MEV 原子套利 | 单桶(5,000 块) '
        f'arb_tx≥{config.BOT_MIN_BUCKET_ARB} 且 '
        f'blocks≥{config.BOT_MIN_BLOCKS} | {len(bot_rows):,} |',
        f'| human | CEX 提币接收方 | 热钱包行为反推 + 探针活跃度排名 | '
        f'{len(human_rows):,}（进采集） |', '',
        f'## 表B 采集与装配层（对应论文表 3.2）', '',
        f'| 组 | 进入采集 | 有记录 | 有效样本（≥{min_n} 条） |',
        '|---|---|---|---|',
    ]
    for g in ('agent', 'bot', 'human'):
        lines.append(f'| {g} | {entered[g]:,} | {got[g]:,} | {written[g]:,} |')
    lines += [f'| 合计 | {sum(entered.values()):,} | {sum(got.values()):,} | '
              f'**{total_eff:,}** |', '',
              '- 去重后不足门槛被剔除: '
              + (', '.join(f'{g}={n}' for g, n in sorted(dropped_thin.items()))
                 or '无'),
              '- 过选余量截断（已达目标后丢弃，非异常）: '
              + (', '.join(f'{g}={n}' for g, n in sorted(n_over_trim.items()))
                 or '无'),
              f'- 触发 {config.MAX_ACTIVITY_PER_ADDRESS:,} 条封顶'
              f'（按段配额保跨度）: {n_capped:,} 个地址',
              f'- 采集覆盖: agent 事件=全窗口连续 / '
              f'bot={len(collect_segs["bot"])} 段'
              f'×{config.BOT_COLLECT_SEG_BLOCKS:,} 块'
              f'（{sum(e - s2 + 1 for _, s2, e in collect_segs["bot"]) / span:.1%}）'
              f' / human={config.HUMAN_COVERAGE:.0%}'
              f'（{len(collect_segs["human"])} 段'
              f'{"，连续流" if len(collect_segs["human"]) == 1 else ""}）'
              f' / olas=全窗口连续',
              '- 🔴 论文交代: 三组均以「窗口内活动量」为入选条件（对称设计），'
              'n_acts 等规模特征已排除在模型外；human 构成偏向常用钱包']
    rpt = config.FIG_DIR / f'SAMPLE_FUNNEL_{chain}.md'
    rpt.write_text('\n'.join(lines), encoding='utf-8')
    log(f'[assemble] 漏斗报告 → {rpt.relative_to(config.ROOT)}', 'ok')

    lvl = 'ok' if total_eff >= config.TARGET_EFFECTIVE_TOTAL else 'warn'
    log(f'[assemble] ✅ 有效样本合计 {total_eff:,}'
        f'（agent {written["agent"]:,} / bot {written["bot"]:,} / '
        f'human {written["human"]:,}）| 目标 '
        f'{config.TARGET_EFFECTIVE_TOTAL:,}', lvl)
    if total_eff < config.TARGET_EFFECTIVE_TOTAL:
        log('  未达标可调: ① sweep_window probe --probe-keep 加大并重跑 '
            'collect/assemble; ② config.BOT_BUCKET_STRIDE=2 扩 bot 池; '
            '③ config.X402_BAD_SUBMITTER_MAX_SHARE 放宽', 'warn')
    return total_eff > 0


# ================================================================
#  CLI
# ================================================================
PHASES = ['hotwallets', 'main', 'mev', 'probe', 'collect', 'assemble',
          'discover', 'activity']


def main():
    ap = argparse.ArgumentParser(
        description='30k 样本方案采集器（全窗口扫描 + 预筛 + 装配）')
    ap.add_argument('phase', choices=PHASES)
    ap.add_argument('--lo', type=int, default=None)
    ap.add_argument('--hi', type=int, default=None)
    ap.add_argument('--blocks', type=int, default=0,
                    help='只用窗口末尾 N 块（快速验证；0=完整建模窗口）')
    ap.add_argument('--force', action='store_true', help='清掉该阶段状态重跑')
    ap.add_argument('--quota', type=int, default=0,
                    help='每组配额覆盖（快速验证用，如 60）')
    ap.add_argument('--probe-keep', type=int, default=0,
                    help='探针排名保留数覆盖（缺省 config.HUMAN_PROBE_KEEP）')
    args = ap.parse_args()

    hi = args.hi or config.MODEL_WINDOW_END_BLOCK
    lo = args.lo or (max(config.MODEL_WINDOW_START_BLOCK, hi - args.blocks)
                     if args.blocks else config.MODEL_WINDOW_START_BLOCK)
    quota = ({g: args.quota for g in config.GROUPS} if args.quota
             else config.GROUP_SELECT_QUOTA)
    keep = args.probe_keep or (args.quota * 4 if args.quota else None)
    # 快速验证模式（--quota）下把候选池同比例缩小 —— 否则探针仍按 15,000 个
    # 候选跑满 360 个 chunk，「快跑」比正式跑还慢，失去验证管线的意义。
    if args.quota:
        config.HUMAN_PROBE_POOL = min(config.HUMAN_PROBE_POOL,
                                      max(args.quota * 50, 1000))
        log(f'  快速模式: human 候选池缩至 {config.HUMAN_PROBE_POOL:,}', 'info')

    log(f'窗口 {lo:,} → {hi:,}（{hi-lo:,} 块 ≈ {(hi-lo)*2/86400:.0f} 天）',
        'step')
    config.SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    steps = {
        'hotwallets': lambda: phase_hotwallets(lo, hi, args.force),
        'main': lambda: phase_main(lo, hi, args.force),
        'mev': lambda: phase_mev(lo, hi, args.force),
        'probe': lambda: phase_probe(lo, hi, args.force, keep),
        'collect': lambda: phase_collect(lo, hi, args.force, quota),
        'assemble': lambda: phase_assemble(lo, hi, quota),
    }
    seq = ({'discover': ['hotwallets', 'main', 'mev', 'probe'],
            'activity': ['collect', 'assemble']}
           .get(args.phase, [args.phase]))
    for s in seq:
        log('', 'info')
        log(f'━━━ 阶段 {s} ━━━', 'step')
        if not steps[s]():
            log(f'阶段 {s} 未完成 —— 修复后重跑同一命令即可续采', 'err')
            sys.exit(1)
    log('全部阶段完成', 'ok')


if __name__ == '__main__':
    main()
