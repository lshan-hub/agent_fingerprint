#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# 🔴 自举：被 sh / dash 调用时，切回 bash 重跑一遍
# ------------------------------------------------------------------------------
# 本脚本用了不少 bash 专有语法：进程替换 >(...)、[[ ]]、数组、${BASH_SOURCE[0]}、
# ${!#} 间接引用。用 `sh run_all.sh` 调用时 sh 以 POSIX 模式解释，
# 第一处进程替换就会报 "syntax error near unexpected token `>'"。
#
# 下面这段必须用纯 POSIX 语法写（[ ] 而非 [[ ]]、$0 而非 $BASH_SOURCE），
# 且必须放在所有 bash 特性之前，否则 sh 解析到那里就已经挂了。
#
# ⚠️ 不能用「检测 BASH_VERSION 是否为空」来判断:
#    macOS 的 /bin/sh 本身就是 bash 3.2，只是以 POSIX 模式运行 ——
#    BASH_VERSION 照样有值（实测 3.2.57），但进程替换被禁用。
#    实测这条检测完全不触发，脚本照样在第 186 行炸掉。
#
# 改用「环境变量标记 + 无条件重启」，不依赖任何模式检测:
#    首次进入 → 标记未设 → 用 bash 重新 exec（argv[0]=bash ⇒ 不进 POSIX 模式）
#    重启之后 → 标记已设 → 正常往下走，不会无限递归
if [ -z "${_AGENTFP_REEXEC:-}" ]; then
  _AGENTFP_REEXEC=1
  export _AGENTFP_REEXEC
  exec bash "$0" "$@"
fi
# ==============================================================================
#  agent_fingerprint —— 一键跑完整条管线（30k 样本方案）
#
#  跑完后 data/ 和 figures/ 的全部内容都会生成:
#
#    data/sources/     各数据源原始产出（olas / x402 / mev / 热钱包 / 画像 ...）
#    data/addresses/   三组真值地址清单（agent_olas / bot_mev / human_cex）
#    data/sweep/       全窗口扫描的分片与断点状态（可随时中断续跑）
#    data/raw/{链}/    统一活动流，按 agent|bot|human 分目录（≈3 万个地址）
#    data/features/    特征宽表 features_{链}.csv + 预测输出
#    data/model/       7 套特征集各自的模型与指标
#    figures/          图表 + go/no-go 判定书 + 样本漏斗报告(SAMPLE_FUNNEL)
#
#  ---------------------------------------------------------------------------
#  用法
#  ---------------------------------------------------------------------------
#    ./bin/run_all.sh                  完整跑（30k 样本；首跑约 1-2 个通宵，
#                                      全程可中断，重跑自动续采）
#    ./bin/run_all.sh --synthetic      合成数据烟测（约 3 分钟，不联网）
#    ./bin/run_all.sh --quick          真实数据快跑（窗口缩到 1 天 + 小配额，
#                                      约 40-60 分钟，只验证管线通不通）
#    ./bin/run_all.sh --from features  跳过采集，从特征开始（数据已有时）
#    ./bin/run_all.sh --force          忽略已有产物，全部重跑
#
#  ---------------------------------------------------------------------------
#  阶段划分（--from 可指定任一阶段作为起点）
#  ---------------------------------------------------------------------------
#    sources    注册表与画像      → data/sources/    首跑约 1 小时（之后全走缓存）
#    discover   ★全窗口发现       → sweep + 三组池子  约 2.5 小时
#               热钱包 → main 趟(x402+Olas exec+提币接收方) → mev 隔桶 → 探针
#    addresses  汇总真值清单      → data/addresses/  约 1 分钟
#    activity   ★采集+装配        → data/raw/        约 10 小时（human 全窗口主导）
#    features   装配特征宽表      → data/features/   约 10-30 分钟（3 万地址）
#    model      训练 7 套对照     → data/model/      约 10-30 分钟
#    figures    出图 + 判定书     → figures/         约 2 分钟
#
#    ⏱ 合计约 23 万次 SQD 请求 ≈ 12 小时（标称 5.2 req/s）。
#      SQD 是免费公共品且长期过载(40~60% 请求返回 529)，实际按 25-37 小时估。
#      → 建议挂通宵；中断随时可续，不会浪费已完成的部分。
#
#    ★ 两个长阶段全部支持断点续跑：中断后重跑本脚本，已完成的 chunk 自动跳过。
#    SQD 免费 Portal 高峰期会持续过载(529)，跑得慢是正常现象，不是卡死 ——
#    日志里能看到 chunk 进度就在推进。
#
#  ---------------------------------------------------------------------------
#  默认是"增量"的
#  ---------------------------------------------------------------------------
#    已经存在的产物会被跳过，不会重复跑。想强制重来加 --force。
#    这样中途断了直接重跑本脚本即可，不会从头再来一遍十几小时的采集。
#
#  ---------------------------------------------------------------------------
#  ⚠️ 关于 30k 样本的口径（论文写作必读）
#  ---------------------------------------------------------------------------
#    · 建模窗口 = config.MODEL_WINDOW_*（2026-03-01 ~ 2026-09-03，187 天），
#      发现与采集都在这个窗口内进行（旧管线的发现窗口在窗口外，已修正）
#    · 有效门槛 ≥20 条活动被前置成入选条件 —— 入选即有效
#    · 三组的入选都以「窗口内活动量」为条件（对称设计），
#      n_acts 等规模特征已排除在模型外
#    · 磁盘预算：data/sweep 峰值约 30-50 GB，最终 data/raw 约 10-15 GB
# ==============================================================================

set -euo pipefail

# 脚本在 bin/ 下，项目根在上一层
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$ROOT"

# 有些系统只有 python3、没有 python
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  command -v python3 >/dev/null 2>&1 && PY=python3 || PY=python
fi

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
SYNTHETIC=""
CHAIN="base"
BLOCKS=0               # 0 = 完整建模窗口（config.MODEL_WINDOW_*）
QUOTA=0                # 0 = config.GROUP_SELECT_QUOTA（13k/11k/6k，合计 30k）
FORCE=0
FROM="sources"
QUICK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --synthetic) SYNTHETIC="--synthetic"; shift ;;
    --chain)     CHAIN="$2"; shift 2 ;;
    --blocks)    BLOCKS="$2"; shift 2 ;;
    --quota)     QUOTA="$2"; shift 2 ;;
    --force)     FORCE=1; shift ;;
    --from)      FROM="$2"; shift 2 ;;
    --quick)     QUICK=1; BLOCKS=43200; QUOTA=60; shift ;;
    -h|--help)
      sed -n '/^#  agent_fingerprint/,/^# ===========/p' "$SELF" \
        | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "未知参数: $1"
      echo "用法: ./bin/run_all.sh [--synthetic] [--quick] [--force]"
      echo "                       [--chain base] [--blocks N] [--quota N]"
      echo "                       [--from sources|discover|addresses|activity|features|model|figures]"
      exit 1 ;;
  esac
done

TAG="$CHAIN"
[[ -n "$SYNTHETIC" ]] && TAG="synthetic"

# sweep_window 的公共参数（窗口 + 配额 + 强制重跑）
SWEEP_ARGS=()
[[ $BLOCKS -gt 0 ]] && SWEEP_ARGS+=(--blocks "$BLOCKS")
[[ $QUOTA -gt 0 ]] && SWEEP_ARGS+=(--quota "$QUOTA")
[[ $FORCE -eq 1 ]] && SWEEP_ARGS+=(--force)

# macOS 自带 bash 3.2：set -u 下展开空数组 "${A[@]}" 会报 unbound variable
# （bash 4.4 才修）。默认跑法三个开关都是 0 → 数组为空 → 阶段 2/4 必炸。
# 调用点一律写 "${SWEEP_ARGS[@]+"${SWEEP_ARGS[@]}"}"：空数组时整体消失，不触发 nounset。

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
STAGES=(sources discover addresses activity features model figures)

# 判断某阶段是否该跑：在 --from 指定的起点之后（含）才跑
stage_enabled() {
  local target="$1" hit=0
  for s in "${STAGES[@]}"; do
    [[ "$s" == "$FROM" ]] && hit=1
    [[ "$s" == "$target" ]] && { [[ $hit -eq 1 ]] && return 0 || return 1; }
  done
  return 1
}

banner() {
  echo
  echo "───────────────────────────────────────────────────────────────"
  echo "  $1"
  echo "───────────────────────────────────────────────────────────────"
}

# 统计 CSV 的「实际数据行数」——排除 # 注释行与表头
#
# 🔴 为什么不能用 -s 判断非空: 本项目的 CSV 都带注释头，
#    一个 0 地址的 bot_mev.csv 仍有 121 字节（2 行注释 + 1 行表头），
#    -s 判它「非空」⇒ 合并被跳过 ⇒ bot 永远补不回来。
csv_data_rows() {
  local p="$1"
  [[ -f "$p" ]] || { echo 0; return; }
  local n
  n=$(grep -vc '^#' "$p" 2>/dev/null || echo 0)
  echo $(( n > 1 ? n - 1 : 0 ))     # 减去表头
}

# 已有产物且没加 --force 就跳过。接受多个路径 —— 必须**全部**存在且非空才跳过。
skip_if_exists() {
  local what="${!#}"                 # 最后一个参数是描述
  local n=$(( $# - 1 ))
  local missing=()
  [[ $FORCE -eq 1 ]] && return 1
  for ((i = 1; i <= n; i++)); do
    local p="${!i}"
    if [[ -d "$p" ]]; then
      [[ -z "$(ls -A "$p" 2>/dev/null)" ]] && missing+=("$p")
    elif [[ "$p" == *.csv ]]; then
      [[ "$(csv_data_rows "$p")" -eq 0 ]] && missing+=("$p")
    elif [[ ! -s "$p" ]]; then
      missing+=("$p")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "  ⏭  已存在，跳过：$what"
    echo "     （想重跑加 --force）"
    return 0
  fi
  return 1
}

# 跑一条命令；失败时打印告警但不中断整条管线（用于可选步骤）。
run_soft() {
  local desc="$1"; shift
  echo "  ▶ $desc"
  if ! "$@"; then
    echo "  ⚠️  失败（不中断）：$desc"
    FAILED_SOFT+=("$desc")
    return 1
  fi
  return 0
}

# 跑一条必须成功的长任务；失败时给出续跑指引并停在这里。
run_hard() {
  local desc="$1"; shift
  echo "  ▶ $desc"
  if ! "$@"; then
    echo
    echo "  🔴 失败：$desc"
    echo "     两个长阶段都支持断点续跑 —— 网络恢复后重跑本脚本即可，"
    echo "     已完成的 chunk 会自动跳过，不会从头再来。"
    exit 1
  fi
  return 0
}

FAILED_SOFT=()
T0=$(date +%s)

# ---------------------------------------------------------------------------
# 运行日志 —— 落在项目内，不要放 /tmp（macOS 会定期清理 /tmp）
# ---------------------------------------------------------------------------
mkdir -p logs
RUN_LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "  运行日志: $RUN_LOG"

# ---------------------------------------------------------------------------
# 目录准备 —— 允许 data/ 和 figures/ 被整个删掉后直接重跑
# ---------------------------------------------------------------------------
mkdir -p data/{sources,addresses,features,model,raw,sweep} figures

# ---------------------------------------------------------------------------
# 开场
# ---------------------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════════"
if [[ -n "$SYNTHETIC" ]]; then
  echo "  合成数据模式 —— 只验证管线通不通，结论无效"
else
  echo "  真实数据模式（30k 样本方案）"
  echo "    链       : $CHAIN"
  if [[ $BLOCKS -gt 0 ]]; then
    echo "    窗口     : 建模窗口末尾 $BLOCKS 块（快速验证，结论不可用于论文）"
  else
    echo "    窗口     : 完整建模窗口 2026-03-01 ~ 2026-09-03（187 天）"
  fi
  if [[ $QUOTA -gt 0 ]]; then
    echo "    每组配额 : $QUOTA（快速验证）"
  else
    echo "    每组配额 : agent 13,000 / bot 11,000 / human 6,000 ⇒ 合计 30,000"
    echo "               （成本感知分配：agent 事件零成本、human 全窗口最贵）"
  fi
  if [[ $QUICK -eq 0 && $BLOCKS -eq 0 ]]; then
    echo "    预计耗时 : 约 23 万次请求 ≈ 12h 标称 / 25-37h 含 SQD 过载"
    echo "               discover ≈2.5h + activity ≈10h（可中断续跑，建议挂通宵）"
    echo "    磁盘预算 : data/sweep 峰值 30-50 GB + data/raw 10-15 GB"
  fi
fi
echo "    起始阶段 : $FROM"
echo "    解释器   : $PY"
[[ $FORCE -eq 1 ]] && echo "    ⚠️  --force：已有产物会被覆盖"
echo "═══════════════════════════════════════════════════════════════"

# ===========================================================================
#  合成模式：直接跳到特征，前面的阶段不适用
# ===========================================================================
if [[ -n "$SYNTHETIC" ]]; then
  banner "[1/3] 生成合成数据 → data/raw/synthetic/"
  "$PY" src/utils/make_synthetic.py

  banner "[2/3] 装配特征 → data/features/features_synthetic.csv"
  "$PY" src/feature/f_ensemble.py --synthetic

  banner "[3/3] 训练 + 出图 → data/model/ + figures/"
  "$PY" src/modeling/train_xgb.py --synthetic
  "$PY" src/modeling/plot_gonogo.py --synthetic

  echo
  echo "═══════════════════════════════════════════════════════════════"
  echo "  合成烟测完成，用时 $(( $(date +%s) - T0 ))s"
  echo "  ⚠️ 合成数据是按 H1 假设生成的，跑出 GO 是必然的 —— 只能证明管线通"
  echo "═══════════════════════════════════════════════════════════════"
  exit 0
fi

# ===========================================================================
#  阶段 1 — 注册表与画像 → data/sources/
# ===========================================================================
if stage_enabled sources; then
  banner "[1/7] 注册表与画像 → data/sources/"
  echo "  Olas/ERC-8004 注册表是全量快照、变动很慢，已有产物直接复用。"
  echo "  x402 提交者画像是 30k 方案的提纯依据（区分微支付网关与零售中继）。"
  echo

  if ! skip_if_exists "data/sources/olas_agents.csv" "Olas 注册表（首抓约 25 分钟）"; then
    run_soft "Olas 自主服务注册表（Base + Gnosis）" \
      "$PY" src/data_fetch/fetch_olas.py --chains base,gnosis || true
  fi
  if ! skip_if_exists "data/sources/erc8004_base.csv" "ERC-8004 注册表"; then
    run_soft "ERC-8004 身份注册表" \
      "$PY" src/data_fetch/fetch_erc8004.py --chain "$CHAIN" --limit 200 || true
  fi
  if ! skip_if_exists "data/sources/x402_facilitators.csv" "x402 提交者画像"; then
    run_soft "x402 提交者画像（6 切片配对金额，约 30 分钟；只需跑一次）" \
      "$PY" src/data_fetch/fetch_x402.py --profile || true
  fi
  run_soft "agenteconomy.to 索引（趋势参考，不参与建模）" \
    "$PY" src/data_fetch/fetch_agenteconomy.py || true

  # Coinglass 需要付费 key，没配就跳过而不是报错
  if "$PY" -c "
import sys; sys.path.insert(0,'.')
from conf import config
sys.exit(0 if (config.COIN_GLASS_KEY or '').strip() else 1)
" 2>/dev/null; then
    run_soft "Coinglass 付费数据源（9 个端点）" \
      "$PY" src/data_fetch/fetch_coinglass.py --all || true
  else
    echo "  ⏭  跳过 Coinglass：conf/config.py 里 COIN_GLASS_KEY 为空"
  fi
else
  echo; echo "⏭  跳过阶段 sources"
fi

# ===========================================================================
#  阶段 2 — ★全窗口发现 → data/sweep/ + 三组池子
# ===========================================================================
if stage_enabled discover; then
  banner "[2/7] 全窗口发现（30k 方案核心）→ 三组真值池"
  echo "  ① hotwallets  窗口内均布切片反推 CEX 热钱包"
  echo "  ② main        ★全窗口一趟：x402 付款方(发现即采集事件活动)"
  echo "                 + Olas Safe 执行事件 + 热钱包提币接收方"
  echo "  ③ mev         隔桶抽样识别原子套利 bot（判据与论文同紧度）"
  echo "  ④ probe       human 候选活跃度探针（只看活动量，不看时间分布）"
  echo
  echo "  每个 chunk 完成即落盘 —— 中断后重跑本脚本自动续采。"
  echo

  run_hard "全窗口发现（hotwallets → main → mev → probe）" \
    "$PY" src/data_fetch/sweep_window.py discover "${SWEEP_ARGS[@]+"${SWEEP_ARGS[@]}"}"
else
  echo; echo "⏭  跳过阶段 discover"
fi

# ===========================================================================
#  阶段 3 — 汇总真值清单 → data/addresses/
# ===========================================================================
if stage_enabled addresses; then
  banner "[3/7] 汇总三组真值地址 → data/addresses/"
  echo "  agent/bot 由 fetch_all 从 data/sources/ 合并去重（含冲突剔除）；"
  echo "  human 已由探针阶段直接写入。"
  echo

  run_soft "合并 agent + bot 清单" "$PY" src/data_fetch/fetch_all.py --only-merge || true

  echo
  echo "  真值清单现状："
  MISSING_GROUPS=()
  for f in agent_olas bot_mev human_cex; do
    if [[ -f "data/addresses/$f.csv" ]]; then
      n=$(csv_data_rows "data/addresses/$f.csv")
      if [[ $n -eq 0 ]]; then
        echo "    $f.csv  ❌ 0 个地址"
        MISSING_GROUPS+=("$f")
      else
        echo "    $f.csv  $n 个地址"
      fi
    else
      echo "    $f.csv  ❌ 缺失"
      MISSING_GROUPS+=("$f")
    fi
  done

  # 🔴 前置闸门 —— 三组缺一不可，缺了就在这里停，别再往下跑十几小时
  if [[ ${#MISSING_GROUPS[@]} -gt 0 ]]; then
    echo
    echo "  🔴 以下真值清单为空或缺失，无法继续："
    for g in "${MISSING_GROUPS[@]}"; do echo "     · $g.csv"; done
    echo
    echo "  三分类任务缺任何一组都不成立，就此停止。常见原因与处理："
    echo "     · discover 阶段未完成 → ./bin/run_all.sh --from discover"
    echo "     · 某个 sweep 阶段失败 → 单独续跑："
    echo "         python3 src/data_fetch/sweep_window.py main"
    echo "         python3 src/data_fetch/sweep_window.py mev"
    echo "         python3 src/data_fetch/sweep_window.py probe"
    exit 1
  fi
else
  echo; echo "⏭  跳过阶段 addresses"
fi

# ===========================================================================
#  阶段 4 — ★采集 + 装配 → data/raw/{chain}/
# ===========================================================================
if stage_enabled activity; then
  banner "[4/7] 采集 + 装配（最长的一步）→ data/raw/$CHAIN/"
  echo "  collect: bot 36 段(4.5%) / human ★全窗口连续 / olas 全窗口"
  echo "  assemble: 配额选样 + 卫生规则 + 写 3 万个地址 JSON + 漏斗报告"
  echo "  ★ agent 组的事件活动已在 discover 的 main 趟采好，这里不再重复扫。"
  echo
  echo "  🔴 human 为什么必须全窗口连续（而不是像 bot 那样分段省时间）："
  echo "     f01 只在同一 seg 内算 Δt，跨段是采样造成的人为断点。"
  echo "     人类天然低频（全窗口约 40 笔），一分段每段就凑不出相邻活动对，"
  echo "     Δt 有效样本 <5 ⇒ 整组在特征阶段被剔除（旧管线只剩 56 个的根因）。"
  echo "  ⇒ 这一步约 8 小时，是本阶段主要耗时，可中断续跑。"
  echo

  run_hard "采集 + 装配（collect → assemble）" \
    "$PY" src/data_fetch/sweep_window.py activity "${SWEEP_ARGS[@]+"${SWEEP_ARGS[@]}"}"

  echo
  echo "  活动流现状："
  for g in agent bot human; do
    d="data/raw/$CHAIN/$g"
    if [[ -d "$d" ]]; then
      echo "    $g  $(ls "$d" | wc -l | tr -d ' ') 个地址"
    else
      echo "    $g  ❌ 缺失"
    fi
  done
  echo "  样本漏斗报告: figures/SAMPLE_FUNNEL_$CHAIN.md（论文表 3.1/3.2 数据源）"
else
  echo; echo "⏭  跳过阶段 activity"
fi

# ===========================================================================
#  阶段 5 — 特征宽表 → data/features/
# ===========================================================================
if stage_enabled features; then
  banner "[5/7] 装配特征宽表 → data/features/features_$TAG.csv"
  echo "  6 族 57 特征：延迟 / 节律 / gas / nonce / 交互 / 金额"
  echo "  3 万个地址约需 10-30 分钟。"
  echo
  "$PY" src/feature/f_ensemble.py --chain "$CHAIN"
else
  echo; echo "⏭  跳过阶段 features"
fi

# ===========================================================================
#  阶段 6 — 训练 7 套对照 → data/model/
# ===========================================================================
if stage_enabled model; then
  banner "[6/7] 训练 7 套特征集对照 → data/model/"
  echo "  从宽到严，每一套堵掉一条"捷径"："
  echo "    model     全部 57 特征                  上界"
  echo "    latency   仅延迟族                      验证 H1"
  echo "    nometa    剔元特征                      堵"协议形态""
  echo "    nowin     剔窗口依赖特征                堵"采集窗口""
  echo "    clean     nometa + nowin                严对照"
  echo "    behavior  再剔 gas/nonce/value 三族     ★★★主结论（堵"缺失模式"）"
  echo "    timing    只剩延迟 + 节律               下界（堵"交互对象"）"
  echo
  "$PY" src/modeling/train_xgb.py --chain "$CHAIN"

  echo
  echo "  ▶ 预测接口冒烟测试（⚠️ 输出是训练集内精度，不是评估结果）"
  "$PY" src/modeling/train_xgb.py --chain "$CHAIN" \
    --predict "data/features/features_$TAG.csv"

  echo
  echo "  ▶ 稳健性检验（三组对照，约 4 分钟）"
  echo "     A 捷径单独成模 / B Δt 截断对照 / C 无延迟族对照"
  echo "     —— 论文 5.6、5.7、5.8 三节的定量依据，缺了会被审稿人打穿"
  run_soft "稳健性检验" "$PY" src/verify/robustness_checks.py --chain "$CHAIN" || true
else
  echo; echo "⏭  跳过阶段 model"
fi

# ===========================================================================
#  阶段 7 — 图表 + 判定书 → figures/
# ===========================================================================
if stage_enabled figures; then
  banner "[7/7] 出图 + go/no-go 判定 → figures/"
  echo "  15 张论文图 + 判定书。判定以最严的 behavior 集为准（与 train_xgb 同口径）。"
  echo
  # 🔴 2026-09-13 修正：此前本阶段只跑 plot_gonogo.py（出 6 张图），
  #    另外 11 张论文图的生成者（gen_missing_figs / regen_doc_figs）
  #    根本没有被接进管线 —— 干净环境下 run_all.sh 跑完必然在
  #    paper_assets.py 处报「缺 9 项论文必需素材」并 exit 1，
  #    与文档宣称的「空目录下一条命令重建全部内容」不符。现补齐。
  echo "  ① go/no-go 四图 + 判定书"
  "$PY" src/modeling/plot_gonogo.py --chain "$CHAIN"
  echo
  echo "  ② 论文配图（图4.1/4.2/5.1 + 图5.8/5.9 的论文口径版本）"
  run_soft "论文配图重生成" "$PY" src/paper/regen_doc_figs.py || true
  echo
  echo "  ③ 补齐图（图4.3/5.2/5.7/5.10/5.11/5.12，含多次拟合，约 1 分钟）"
  run_soft "补齐论文图" "$PY" src/paper/gen_missing_figs.py || true
else
  echo; echo "⏭  跳过阶段 figures"
fi

# ===========================================================================
#  收尾
# ===========================================================================
ELAPSED=$(( $(date +%s) - T0 ))
echo
echo "═══════════════════════════════════════════════════════════════"
echo "  完成，用时 $(( ELAPSED / 3600 )) 小时 $(( ELAPSED % 3600 / 60 )) 分"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "  data/ 产出"
for d in sources addresses "raw/$CHAIN" features model; do
  if [[ -d "data/$d" ]]; then
    printf "    %-16s %s\n" "$d" \
      "$(find "data/$d" -type f | wc -l | tr -d ' ') 个文件, $(du -sh "data/$d" 2>/dev/null | cut -f1)"
  fi
done
echo
echo "  figures/ 产出"
for f in figures/*"$TAG"*; do
  [[ -e "$f" ]] && printf "    %s\n" "$(basename "$f")"
done
echo
echo "  重点看这五份："
echo "    figures/PAPER_ASSETS_$TAG.md     论文图表对照清单（15 图 + 7 表）"
echo "    figures/SAMPLE_FUNNEL_$TAG.md    样本漏斗（论文表 3.1/3.2 数据源）"
echo "    figures/GONOGO_VERDICT_$TAG.md   go/no-go 判定书"
echo "    figures/TRAIN_REPORT_$TAG.md     7 套特征集对照 + 四类捷径分析"
echo "    figures/ROBUSTNESS_$TAG.md       ★稳健性检验（论文 5.6/5.7/5.8 的定量依据）"
echo
echo "  💡 data/sweep/ 是采集中间产物（30-50 GB），全部跑通并验收后可删除。"

if [[ ${#FAILED_SOFT[@]} -gt 0 ]]; then
  echo
  echo "  ⚠️  以下步骤失败了（已跳过，不影响下游）："
  for d in "${FAILED_SOFT[@]}"; do echo "     · $d"; done
fi
echo

# ===========================================================================
#  产物验收 —— 跑完不报错 ≠ 结果对
# ===========================================================================
if stage_enabled figures; then
  banner "产物验收"
  if "$PY" src/verify/verify_outputs.py --chain "$CHAIN"; then
    VERIFY_RC=0
  else
    VERIFY_RC=$?
  fi
  if [[ ${VERIFY_RC:-0} -ne 0 ]]; then
    echo
    echo "  🔴 验收未通过 —— 上面列出的问题需要处理后重跑。"
    exit 1
  fi

  # 论文素材核对 —— 逐条确认论文要用的每张图、每张表都已产出。
  # 与 verify_outputs 互补: 那个验「管线产物是否合理」，这个验「论文要的齐不齐」。
  banner "论文素材核对"
  if ! "$PY" src/paper/paper_assets.py --chain "$CHAIN"; then
    echo
    echo "  🔴 论文素材不齐 —— 见上面的缺失清单。"
    exit 1
  fi
fi
echo
