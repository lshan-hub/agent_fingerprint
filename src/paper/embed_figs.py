#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
@Project ：agent_fingerprint
@File    ：embed_figs.py
@Description:
    把 figures/ 下的最新 PNG 以 base64 重新内嵌进论文 HTML。

    背景：论文是自包含单文件（图片全部 base64 内嵌，便于提交与分发），
    因此数据重跑后必须显式把新图替换进去 —— 否则正文数字已更新、
    配图仍是旧的，是最难被发现的一类不一致。

    匹配方式：按 <img ... alt="图 X.Y ..."> 里的图号定位，而不是按出现
    顺序 —— 顺序会随论文编辑漂移，图号不会。

    ⚠️ 图 5.12 的 alt 文案在本轮由「三个类别各自的代表性决策树」改为
       「单棵提升树的结构展开」，映射表里按图号匹配，不受文案影响。
'''

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'doc' / '链上AI_Agent账户行为指纹识别的研究.html'
FIG = ROOT / 'figures'

# 图号 → figures/ 下的文件名
MAP = {
    '4.1':  'fig41_feature_system.png',
    '4.2':  'fig42_band_burst_def.png',
    '4.3':  'fig43_missingness.png',
    '5.1':  'fig_model_arch.png',
    '5.2':  'fig_ablation_frame.png',
    '5.3':  'fig1_latency_distribution_base.png',
    '5.4':  'fig2_fast_response_ecdf_base.png',
    '5.5':  'fig3_band_shares_base.png',
    '5.6':  'fig4_burstiness_entropy_base.png',
    '5.7':  'fig7_learning_curve_base.png',
    # 🔴 5.8 用 behavior 集的混淆矩阵（主结论口径），不是 plot_gonogo 的
    #    全特征版 fig5_confusion_base.png —— 正文讨论的 498 例误分类
    #    出自 behavior 集，配错图会与数字对不上。
    '5.8':  'fig5_confusion_behavior_base.png',
    # 同理，5.9 用增益版重要性而非不纯度版
    '5.9':  'fig6_importance_gain_base.png',
    '5.10': 'fig9_oof_proba_base.png',
    '5.11': 'fig10_roc_ovr_base.png',
    '5.12': 'fig8_tree_structure_base.png',
}


def main():
    html = DOC.read_text(encoding='utf-8')
    n_ok = n_miss = 0

    def repl(m):
        nonlocal n_ok, n_miss
        head, alt, tail = m.group(1), m.group(2), m.group(3)
        num = re.match(r'图\s*([0-9.]+)', alt)
        if not num:
            return m.group(0)
        key = num.group(1).rstrip('.')
        fn = MAP.get(key)
        if not fn:
            print(f'  ⚠ 图{key} 无映射，跳过')
            n_miss += 1
            return m.group(0)
        p = FIG / fn
        if not p.exists():
            print(f'  ❌ 图{key} 源文件缺失: {fn}')
            n_miss += 1
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode()
        n_ok += 1
        print(f'  ✓ 图{key:<5} ← {fn}  ({p.stat().st_size/1024:.0f} KB)')
        return f'{head}data:image/png;base64,{b64}{tail}'

    pat = re.compile(r'(<img src=")data:image/png;base64,[^"]*("[^>]*?alt=")',
                     re.S)
    # 两段式替换：先定位 alt 才能知道该换成哪张图
    out, pos, buf = [], 0, html
    for m in re.finditer(
            r'(<img src=")data:image/png;base64,[^"]*("[^>]*alt=")([^"]*)"',
            buf, re.S):
        head, mid, alt = m.group(1), m.group(2), m.group(3)
        num = re.match(r'图\s*([0-9.]+)', alt)
        key = num.group(1).rstrip('.') if num else None
        fn = MAP.get(key) if key else None
        p = (FIG / fn) if fn else None
        out.append(buf[pos:m.start()])
        if p and p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            out.append(f'{head}data:image/png;base64,{b64}{mid}{alt}"')
            n_ok += 1
            print(f'  ✓ 图{key:<5} ← {fn}  ({p.stat().st_size/1024:.0f} KB)')
        else:
            out.append(m.group(0))
            print(f'  ❌ 图{key} 未替换（映射或文件缺失）')
            n_miss += 1
        pos = m.end()
    out.append(buf[pos:])
    DOC.write_text(''.join(out), encoding='utf-8')
    print(f'\n完成：替换 {n_ok} 张，跳过 {n_miss} 张')
    print(f'文件大小: {DOC.stat().st_size/1024:.0f} KB')
    return 0 if n_miss == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
