#!/usr/bin/env bash
# 管线看门狗：阻塞等待"有意义的事件"，命中即退出并说明原因。
# 用途：把"每 10 分钟轮询一次"换成"只在该看的时候叫我"。
#
# 触发条件（任一命中即退出）：
#   1. 日志出现新的阶段横幅 [N/7]        → 阶段切换
#   2. 日志出现 🔴 / ❌ / Traceback      → 出错
#   3. 日志连续 IDLE_MAX 秒没有增长       → 疑似卡死
#   4. 主任务进程已退出                   → 跑完或挂了
#   5. 达到 MAX_WAIT 秒上限               → 例行检查点
#
# 用法: bash bin/_watch.sh <日志路径> [MAX_WAIT秒]

LOG="$1"
MAX_WAIT="${2:-10800}"      # 缺省最长等 3 小时
IDLE_MAX=3600               # 60 分钟无输出才视为停滞
                            # 🔴 不能设更短：main/activity 阶段的进度行每 20 个 chunk
                            #    才打一次，SQD 过载时 20 个 chunk 要走 20-40 分钟，
                            #    日志静默是常态而非故障。判活要看 data/sweep 是否
                            #    增长 + 进程 CPU 时间是否推进（实测 6 分钟 +7.1MB /
                            #    +19s CPU，日志却一个字节没动）。设 1200 会持续误报。
POLL=20

[ -f "$LOG" ] || { echo "WATCH: 日志不存在 $LOG"; exit 2; }

t0=$(date +%s)
last_size=$(wc -c < "$LOG")
last_sweep=$(du -sk "$(dirname "$0")/../data/sweep" 2>/dev/null | cut -f1)
last_sweep="${last_sweep:-0}"
last_change=$t0
base_stage=$(grep -oE '\[[0-9]/7\]' "$LOG" | tail -1)

while :; do
  sleep "$POLL"
  now=$(date +%s)
  size=$(wc -c < "$LOG")

  # —— 主进程还在吗
  if ! pgrep -f "run_all.sh" > /dev/null 2>&1; then
    echo "WATCH: 主任务已退出（跑完或中断）"
    tail -25 "$LOG"; exit 0
  fi

  # —— 出错
  # 🔴 只匹配「真失败」措辞。不能裸匹配 🔴/失败 ——
  #    管线日志把 🔴 当作「重点说明」的标记大量使用（如
  #    "🔴 提醒：Olas agent 是 Gnosis Safe"、"🔴 human 为什么必须全窗口连续"），
  #    裸匹配会在阶段 1 就误报。实测已踩。
  if tail -60 "$LOG" | grep -qE 'Traceback \(most recent|🔴 失败|🔴 验收未通过|🔴 以下真值清单|🔴 论文素材不齐|⚠️  失败（不中断）|^\s*❌ '; then
    echo "WATCH: 日志中出现错误标记"
    tail -40 "$LOG"; exit 3
  fi

  # —— 阶段切换
  cur_stage=$(grep -oE '\[[0-9]/7\]' "$LOG" | tail -1)
  if [ "$cur_stage" != "$base_stage" ] && [ -n "$cur_stage" ]; then
    echo "WATCH: 阶段切换 $base_stage → $cur_stage"
    tail -25 "$LOG"; exit 0
  fi

  # —— 停滞
  # 判活取「日志字节」与「data/sweep 落盘量」的并集：任一增长即视为存活。
  # 只看日志会误报（见 IDLE_MAX 注释）；只看 sweep 则在非采集阶段失效。
  sweep_kb=$(du -sk "$(dirname "$0")/../data/sweep" 2>/dev/null | cut -f1)
  sweep_kb="${sweep_kb:-0}"
  if [ "$size" != "$last_size" ] || [ "$sweep_kb" != "$last_sweep" ]; then
    last_size=$size; last_sweep=$sweep_kb; last_change=$now
  elif [ $(( now - last_change )) -ge "$IDLE_MAX" ]; then
    echo "WATCH: 日志与 data/sweep 已 $(( (now - last_change) / 60 )) 分钟同时无增长，疑似停滞"
    tail -25 "$LOG"; exit 4
  fi

  # —— 例行上限
  if [ $(( now - t0 )) -ge "$MAX_WAIT" ]; then
    echo "WATCH: 达到 $(( MAX_WAIT / 60 )) 分钟检查点（管线仍在正常推进）"
    tail -15 "$LOG"; exit 0
  fi
done
