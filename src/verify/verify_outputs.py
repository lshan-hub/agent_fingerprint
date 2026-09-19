#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：verify_outputs.py
@Description:
    产物验收 —— 跑完 run_all.sh 后，逐条核对 data/ 与 figures/ 是否符合预期。

    ============================================================
    一、 为什么需要这个脚本
    ============================================================
    run_all.sh 跑完不报错 ≠ 结果对。实测踩过的坑:
      · 地址清单里混进 3,560 个 Gnosis 地址，在 Base 上永远采不到活动
      · Coinglass 的 401 被吞成「成功 0 行」，看日志以为通了
      · 采集窗口不足一周时 entropy_dow 恒为 0，变成纯窗口伪影
      · 某组全部地址活动数不足，整组在特征阶段被剔除
    这些都不会让脚本退出码非零，只能靠对产物本身做断言来发现。

    ============================================================
    二、 检查分五层，逐层递进
    ============================================================
      ① 文件存在性   该有的目录和文件都在，且非空
      ② 数据完整性   三组都有样本，数量达到可建模的下限
      ③ 科学有效性   窗口够一周、跨度守住、无跨链污染
      ④ 模型合理性   7 套特征集齐全，主结论指标过线
      ⑤ 图表产出     6 张图 + 2 份报告，且文件大小合理

    ============================================================
    三、 退出码
    ============================================================
      0 = 全部通过（可能有 warn）
      1 = 有 FAIL —— 结果不可用，需要重跑或修代码
'''

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conf import config  # noqa: E402


class OutputVerifier(object):
    """
    产物验收器

    核心函数 (共 6 个):
      ① check_files()     文件存在性
      ② check_data()      数据完整性
      ③ check_science()   科学有效性（窗口/跨度/跨链）
      ④ check_model()     模型合理性
      ⑤ check_figures()   图表产出
      ⑥ run()             ①~⑤ + 汇总判定

    Args:
        chain: 链名，缺省 config.PRIMARY_CHAIN
    """

    # 每组最少要有多少个样本才够建模（低于此值 CV 折数都凑不齐）
    MIN_SAMPLES_PER_GROUP = 25
    # 主结论（behavior 集）的及格线，与 config.GONOGO_* 对齐
    MIN_BEHAVIOR_F1 = 0.70
    MIN_AGENT_RECALL = 0.60
    # 图表文件至少这么大才算画出了东西（空白图约 10~20KB）
    MIN_FIG_BYTES = 25_000

    EXPECTED_FEATURE_SETS = ['model', 'latency', 'nometa', 'nowin',
                             'clean', 'behavior', 'timing']
    EXPECTED_FIGS = ['fig1_latency_distribution', 'fig2_fast_response_ecdf',
                     'fig3_band_shares', 'fig4_burstiness_entropy',
                     'fig5_confusion', 'fig6_importance']

    def __init__(self, chain: str = None):
        self.chain = chain or config.PRIMARY_CHAIN
        self.fails = []
        self.warns = []
        self.oks = []
        # 各组首末活动跨度中位（天）—— 全量统计，论文 4.5 节直接引用
        self.spans_days = {}

    # ============================================================
    #  断言小工具
    # ============================================================
    def ok(self, msg: str):
        self.oks.append(msg)
        print(f'  ✅ {msg}')

    def fail(self, msg: str):
        self.fails.append(msg)
        print(f'  ❌ {msg}')

    def warn(self, msg: str):
        self.warns.append(msg)
        print(f'  ⚠️  {msg}')

    def expect(self, cond: bool, ok_msg: str, fail_msg: str):
        """断言：真则记 ok，假则记 fail。"""
        self.ok(ok_msg) if cond else self.fail(fail_msg)
        return cond

    # ============================================================
    #  ① 文件存在性
    # ============================================================
    def check_files(self) -> None:
        """① 该有的目录和文件都在，且非空。"""
        print('\n① 文件存在性')
        for d in (config.DATA_SRC, config.ADDR_DIR, config.DATA_RAW,
                  config.DATA_FEAT, config.DATA_DIR / 'model', config.FIG_DIR):
            n = len(list(d.rglob('*'))) if d.exists() else 0
            self.expect(d.exists() and n > 0,
                        f'{d.name}/ 存在且非空（{n} 项）',
                        f'{d.name}/ 缺失或为空 —— 该阶段没有产出')

        for g in config.GROUPS:
            f = config.ADDR_DIR / config.GROUPS[g][0]
            self.expect(f.exists() and f.stat().st_size > 100,
                        f'{f.name} 已生成',
                        f'{f.name} 缺失 —— addresses 阶段失败')

    # ============================================================
    #  ② 数据完整性
    # ============================================================
    def check_data(self) -> None:
        """② 三组都有样本，且数量够建模。"""
        print('\n② 数据完整性')
        for g in config.GROUPS:
            d = config.DATA_RAW / self.chain / g
            n = len(list(d.glob('*.json'))) if d.exists() else 0
            self.expect(n > 0, f'raw/{self.chain}/{g} 采到 {n} 个地址',
                        f'raw/{self.chain}/{g} 为空 —— 该组采集失败')

        p = config.DATA_FEAT / f'features_{self.chain}.csv'
        if not self.expect(p.exists(), '特征宽表已生成',
                           '特征宽表缺失 —— features 阶段失败'):
            return
        df = pd.read_csv(p)
        vc = df['group'].value_counts()
        for g in config.GROUPS:
            n = int(vc.get(g, 0))
            if n >= self.MIN_SAMPLES_PER_GROUP:
                self.ok(f'{g} 组 {n} 个样本（下限 {self.MIN_SAMPLES_PER_GROUP}）')
            elif n > 0:
                self.warn(f'{g} 组仅 {n} 个样本，低于建模下限 '
                          f'{self.MIN_SAMPLES_PER_GROUP} —— 结论会不稳')
            else:
                self.fail(f'{g} 组 0 个样本 —— 整组在特征阶段被剔除了')
        self.df = df

    # ============================================================
    #  ③ 科学有效性
    # ============================================================
    def check_science(self) -> None:
        """
        ③ 窗口够一周、跨度守住、无跨链污染

        这三项是「跑完不报错但结论无效」的高发区，必须单独验。
        """
        print('\n③ 科学有效性')

        # 3.1 跨链污染 —— 地址清单里不该有别的链的地址被采进来
        known = set(config.SQD_DATASETS) | set(config.CHAINS)
        others = {c for c in known if c != self.chain}
        bad = 0
        for g in config.GROUPS:
            f = config.ADDR_DIR / config.GROUPS[g][0]
            if not f.exists():
                continue
            raw = [l for l in f.read_text(encoding='utf-8').splitlines()
                   if l and not l.startswith('#')]
            if len(raw) < 2:
                continue
            import csv as _csv
            for r in _csv.DictReader(raw):
                src = (r.get('source') or '').lower()
                if any(c in src for c in others):
                    bad += 1
        if bad:
            self.warn(f'地址清单含 {bad} 个非 {self.chain} 链地址 —— '
                      f'采集端 _filter_by_chain 会剔除，不影响结果')
        else:
            self.ok(f'地址清单无跨链污染')

        # 3.2 采集窗口 —— 周期类特征需要跨度 >= 7 天
        # 🔴 2026-09-13 修正：此前写的是 list(d.glob('*.json'))[:40]，
        #    只读前 40 个文件（按地址十六进制排序，纯便利样本）就报"中位数"。
        #    实测偏差极大：agent 组 40 样本中位 27.5 天 vs 全量 18,305 个
        #    地址的真实中位 13.3 天 —— 差了 2.07 倍，而论文 4.5 节正是引用
        #    这行的输出。便利样本的统计量不得当作总体统计量对外报告。
        #    改为全量统计（3 万个 JSON 约 20 秒，可接受）。
        need = config.MIN_WINDOW_FOR_CYCLE_BLOCKS
        for g in config.GROUPS:
            d = config.DATA_RAW / self.chain / g
            fs = sorted(d.glob('*.json')) if d.exists() else []
            if not fs:
                continue
            spans = []
            for fp in fs:
                try:
                    r = json.loads(fp.read_text())
                except json.JSONDecodeError:
                    continue
                if len(r) > 1:
                    bn = [x['blockNumber'] for x in r]
                    spans.append(max(bn) - min(bn))
            if not spans:
                continue
            spans.sort()
            n = len(spans)
            med = (spans[n // 2] if n % 2
                   else (spans[n // 2 - 1] + spans[n // 2]) // 2)
            days = med * config.SQD_BLOCK_TIME_MEASURED.get(self.chain, 2.0) / 86400
            self.spans_days[g] = round(days, 1)
            if med >= need * 0.9:
                self.ok(f'{g} 组区块跨度中位 {med:,} 块 ≈ {days:.1f} 天'
                        f'（全量 {n:,} 个地址；需 ≥7 天）')
            else:
                # 这里说的是「该组地址首末活动的实际间隔」，不是采集窗口。
                # human 组天然低频，活动稀疏 ⇒ 首末间隔本就短于采集窗口，
                # 这是数据的固有属性，不是采集配置错误。
                # 主结论不受影响: entropy_dow / entropy_24h / active_hours
                # 都已列入 WINDOW_DEP_FEATURES，BEHAVIOR_FEATURES 已剔除它们。
                self.warn(
                    f'{g} 组首末活动跨度中位仅 {med:,} 块 ≈ {days:.1f} 天，'
                    f'不足一周 —— 该组周期类特征（entropy_dow 等）不可靠。'
                    f'⚠️ 但主结论用的 behavior 集已剔除这些特征，结论不受影响；'
                    f'论文写作时不要引用该组的 entropy_dow')

        # 3.3 各组窗口是否等长（不等长则周期类特征不可比）
        if hasattr(self, 'df') and 'win_blocks' in self.df.columns:
            wb = self.df.groupby('group')['win_blocks'].median()
            if len(wb) > 1 and wb.max() / max(wb.min(), 1) > 1.5:
                self.fail(f'各组窗口长度不等 {dict(wb.astype(int))} —— '
                          f'周期类特征不可比')
            else:
                self.ok('各组采集窗口等长，周期类特征可比')

    # ============================================================
    #  ④ 模型合理性
    # ============================================================
    def check_model(self) -> None:
        """④ 7 套特征集齐全，主结论过线。"""
        print('\n④ 模型合理性')
        mdir = config.DATA_DIR / 'model'
        got = {}
        for name in self.EXPECTED_FEATURE_SETS:
            p = mdir / f'{self.chain}_{name}' / 'metrics.json'
            if not p.exists():
                self.fail(f'特征集 {name} 未训练（缺 {p.parent.name}/metrics.json）')
                continue
            try:
                got[name] = json.loads(p.read_text())
            except json.JSONDecodeError:
                self.fail(f'特征集 {name} 的 metrics.json 解析失败')
        if len(got) == len(self.EXPECTED_FEATURE_SETS):
            self.ok(f'7 套特征集全部训练完成')

        # 主结论以 behavior 为准
        b = got.get('behavior')
        if not b:
            self.fail('主结论特征集 behavior 缺失 —— 无法判定 go/no-go')
            return
        f1 = b.get('oof_macro_f1', 0)
        rec = (b.get('oof_recall') or {}).get('agent', 0)
        self.expect(f1 >= self.MIN_BEHAVIOR_F1,
                    f'behavior macro-F1 = {f1:.4f}（线 {self.MIN_BEHAVIOR_F1}）',
                    f'behavior macro-F1 = {f1:.4f} 未过线 {self.MIN_BEHAVIOR_F1}')
        self.expect(rec >= self.MIN_AGENT_RECALL,
                    f'behavior agent 召回 = {rec:.4f}（线 {self.MIN_AGENT_RECALL}）',
                    f'behavior agent 召回 = {rec:.4f} 未过线 {self.MIN_AGENT_RECALL}')

        # 捷径体检：全特征与 behavior 差太多说明判别力来自捷径而非行为
        m = got.get('model')
        if m:
            d = m.get('oof_macro_f1', 0) - f1
            if d > 0.15:
                self.warn(f'全特征比 behavior 高 {d:+.4f} —— '
                          f'判别力大部分来自协议/窗口/缺失模式这些捷径')
            elif m.get('oof_macro_f1', 0) >= 0.99:
                # 🔴 天花板区间的 Δ 没有分辨率，不能读成「捷径贡献小」。
                #    真正的判据是捷径单独成模能到多少（src/verify/robustness_checks.py）。
                self.ok(f'全特征与 behavior 差值 {d:+.4f}（⚠️ model 集已达 '
                        f'{m["oof_macro_f1"]:.4f}，处于天花板区间：该差值只说明'
                        f'捷径互相冗余，不可解读为「捷径贡献小」—— '
                        f'见 ROBUSTNESS 报告表 A）')
            else:
                self.ok(f'全特征与 behavior 差值 {d:+.4f}，捷径边际贡献有限')

        # 模型文件本身
        for name in got:
            f = mdir / f'{self.chain}_{name}' / 'xgb_clf_full.pkl'
            if not (f.exists() and f.stat().st_size > 1000):
                self.fail(f'{name} 的模型文件缺失或过小')

    # ============================================================
    #  ⑤ 图表产出
    # ============================================================
    def check_figures(self) -> None:
        """⑤ 6 张图 + 2 份报告，且不是空白图。"""
        print('\n⑤ 图表产出')
        for stem in self.EXPECTED_FIGS:
            p = config.FIG_DIR / f'{stem}_{self.chain}.png'
            if not p.exists():
                self.fail(f'{p.name} 缺失')
            elif p.stat().st_size < self.MIN_FIG_BYTES:
                self.warn(f'{p.name} 仅 {p.stat().st_size/1024:.0f}KB，'
                          f'可能是空白图')
            else:
                self.ok(f'{p.name}（{p.stat().st_size/1024:.0f}KB）')

        for name in (f'GONOGO_VERDICT_{self.chain}.md',
                     f'TRAIN_REPORT_{self.chain}.md'):
            p = config.FIG_DIR / name
            self.expect(p.exists() and p.stat().st_size > 500,
                        f'{name} 已生成', f'{name} 缺失或过短')

    # ============================================================
    #  ⑥ 主流程
    # ============================================================
    def run(self) -> int:
        """⑥ 跑完五层检查并汇总。Returns: 退出码（0 通过 / 1 有 FAIL）。"""
        print('=' * 66)
        print(f'  产物验收 —— 链: {self.chain}')
        print('=' * 66)

        self.check_files()
        self.check_data()
        self.check_science()
        self.check_model()
        self.check_figures()

        print('\n' + '=' * 66)
        print(f'  通过 {len(self.oks)} 项 | 警告 {len(self.warns)} 项 | '
              f'失败 {len(self.fails)} 项')
        print('=' * 66)
        if self.fails:
            print('\n❌ 未通过的检查:')
            for f in self.fails:
                print(f'   · {f}')
            print('\n结果不可用。修掉上面的问题后重跑 ./bin/run_all.sh')
            return 1
        if self.warns:
            print('\n⚠️  提示（不影响可用性，但要在论文里交代）:')
            for w in self.warns:
                print(f'   · {w}')
        print('\n✅ 全部产物符合预期。')
        return 0


def main():
    import argparse
    ap = argparse.ArgumentParser(description='验收 run_all.sh 的产物')
    ap.add_argument('--chain', default=config.PRIMARY_CHAIN)
    args = ap.parse_args()
    sys.exit(OutputVerifier(chain=args.chain).run())


if __name__ == '__main__':
    main()
