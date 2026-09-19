from pathlib import Path

# ============================================================
# 路径
# ============================================================
# 本文件在 conf/ 下，所以项目根是父目录的父目录
ROOT = Path(__file__).resolve().parent.parent

CONF_DIR = ROOT / "conf"
SRC_DIR = ROOT / "src"
DOC_DIR = ROOT / "doc"

DATA_DIR = ROOT / "data"
ADDR_DIR = DATA_DIR / "addresses"       # 三组真值地址清单
DATA_RAW = DATA_DIR / "raw"             # 拉下来的原始交易
DATA_FEAT = DATA_DIR / "features"       # 算好的特征表
DATA_SRC = DATA_DIR / "sources"         # data_fetch 各数据源的产出
FIG_DIR = ROOT / "figures"

for _d in (ADDR_DIR, DATA_RAW, DATA_FEAT, DATA_SRC, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🥇 主力数据源 —— SQD Network 公开 Portal
# ============================================================

SQD_PORTAL = "https://portal.sqd.dev"
SQD_DATASET = "base-mainnet"
SQD_USER_AGENT = "agent-fingerprint-research/0.1"

# SQD 共 139 个数据集，全部 start_block=0（全历史）。本课题相关的链映射：
# ⚠️ BSC 的数据集名是 binance-mainnet，不是 bsc-mainnet（试了 6 个名字才找到）
SQD_DATASETS = {
    "ethereum": "ethereum-mainnet",
    "base": "base-mainnet",
    "bsc": "binance-mainnet",      # ← 注意命名
    "arbitrum": "arbitrum-one",
    "optimism": "optimism-mainnet",
    "gnosis": "gnosis-mainnet",
    "polygon": "polygon-mainnet",
    "solana": "solana-mainnet",
}

SQD_BLOCK_TIME_MEASURED = {
    "bsc": 1.0, "arbitrum": 1.0, "base": 2.0, "gnosis": 5.0, "ethereum": 12.0,
}

# 实测：8 并发最优 5.22 req/s（24/24 成功）；16 并发降到 3.09 req/s 且 32/48 失败
SQD_WORKERS = 8
SQD_MAX_RETRIES = 6

# 🔴 网络类错误（DNS/连接重置/超时）的独立重试预算 —— 2026-09-05 实测新增
# 基础预算 6 次的退避是 1+2+4+8+16+32 = 63 秒，实测一次几分钟的网络中断
# 就打穿了它，导致 MEV 与 human 两组采集失败、地址清单为空、
# activity 阶段跳过、最终只剩 agent 一个类别、7 套模型全部训练失败。
# 10 次的退避是 1,2,4,8,16,32,64,120,120,120 ≈ 8 分钟。
SQD_NET_RETRIES = 10
SQD_CHUNK_BLOCKS = 396          # 实测单次响应的扫描窗口，仅作进度估算用
SQD_MAX_ADDR_PER_QUERY = 1000   # 实测 1000 个地址可一次过滤（3.9s）
SQD_BISECT_ROUNDS = 14

# ★ 字段配置：裁掉 input、保留 sighash
#   实测全窗口体积：含 input 4.02TB / 裁光 0.67TB / 保留 sighash 0.76TB
#   sighash 只多 11GB，却是 method_entropy 特征的唯一来源 —— 必须留
SQD_FIELDS = {
    "transaction": {
        "from": True, "to": True, "hash": True, "nonce": True,
        "gasPrice": True, "gasUsed": True, "value": True,
        "status": True, "transactionIndex": True, "sighash": True,
    },
    "block": {"number": True, "timestamp": True},
}

# 建模时间窗（实测二分定位）
WINDOW_START_DATE = "2025-09-01"    # x402 发布，叙事起点
WINDOW_START_BLOCK = 34_942_382     # 实测：Base 该日期对应高度
WINDOW_END_BLOCK = 50_800_000       # 2026-09-03
# 全窗口 15,857,618 个区块 → 全网扫描约 40,044 次请求 / 2.1 小时 / 95GB Parquet


# ============================================================
# 💰 付费数据源 —— CoinGlass Open API v4
# ============================================================
# ⚠️ API Key 定义在本文件末尾的「Coinglass付费服务」段落（COIN_GLASS_KEY）

COIN_GLASS_BASE_URL = "https://open-api-v4.coinglass.com/api"

# 档位 → 每分钟请求上限。环境变量 COINGLASS_TIER 指定，缺省 hobbyist
COIN_GLASS_TIER = "hobbyist"
COIN_GLASS_TIER_RPM = {
    "hobbyist": 30,        # $29/月
    "startup": 80,         # $79/月
    "standard": 300,       # $299/月
    "professional": 1200,  # $699/月
}

# CoinGlass 币种名映射（API 路径用全小写英文名，非 ticker）
COIN_GLASS_SYMBOL_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "xrp",
}


# ============================================================
# ★ 建模时间窗口（任务1 结论，2026-09-03 实测定标）
# ============================================================
# 【实测密度】每 3000 区块的 x402 事件 / 去重付款方 / 人均笔数：
#     2025-12 峰值期   99,871 / 5,847 / 17.1
#     2026-03 崩塌后    6,237 / 1,225 /  5.1
#     2026-09 当前     11,087 / 4,053 /  2.7
#   ★ 峰值期人均 17.1 笔 vs 当前 2.7 笔 —— 强度降 6.3 倍但地址数只降 31%
#     ⇒ 投机退潮、真实用户留存。这是论文核心叙事的直接数据证据。
#
# 【为什么选 2026-03 起的 6 个月】
#   1. 崩塌后的稳态期 —— 这里的 agent 是泡沫退潮后留存的真实自动化
#   2. ERC-8004 于 2026-01-29 上主网，早于窗口起点 ⇒ 全覆盖
#   3. 样本充足 —— 实测每 3000 区块有 1,225~4,053 个去重付款方
#   4. 每地址有 6 个月活动史，足以跨过 MIN_TX_FOR_FEATURES 门槛
MODEL_WINDOW_START_DATE = "2026-03-01"
MODEL_WINDOW_START_BLOCK = 42_755_201
MODEL_WINDOW_END_BLOCK = 50_819_862      # 2026-09-03
# 跨度 8,064,661 区块 ≈ 187 天

# 对照窗口（峰值期采样，仅用于「投机 vs 真实」的叙事对照，不参与训练）
BUBBLE_WINDOW_START_BLOCK = 38_865_534   # 2025-12-01
BUBBLE_WINDOW_BLOCKS = 100_000           # 约 2.3 天

# 🔴 每地址活动条数上限 —— 不封顶会爆炸
#   实测：单个 MEV bot 的 nonce 高达 2,911,255，6 个月约 525 MB
#         700 个 bot 就是 359 GB，远超 5 GB 预算
#   封顶 3000 条后：1,900 地址 × 3,000 × 250B = 1.33 GB ✅
#   行为特征只需几千条就能算出稳定统计量，封顶不损失信息
MAX_ACTIVITY_PER_ADDRESS = 3000


# ============================================================
# 🔴 统一活动流口径（任务1 实测发现，必须这样做）
# ============================================================
# 实测三组的链上足迹形态完全不同：
#     组             是合约   nonce(自己发出的交易数)
#     Olas Safe      ✅ 是    1          ← 只有部署那一笔
#     x402 付款方     否      2 ~ 1,079  ← 走元交易，Facilitator 代付
#     MEV bot        否      749,745 ~ 2,911,255
#
# ⇒ 用 tx.from 采集，Olas 和 x402 几乎采不到任何数据（实测 6000 区块内为 0）
#   而 x402 的活动 100% 体现在 AuthorizationUsed 事件里（实测 8 地址 7,591 条）
#
# ★ 正确口径：Δt = 该地址相邻两次【链上活动】的间隔，活动包括
#     ① 作为 tx.from  发出交易        （MEV bot / 人类 的主要形态）
#     ② 作为 tx.to    被调用          （Olas Safe 的主要形态）
#     ③ 出现在事件参数（AuthorizationUsed.authorizer / Transfer.from）
#                                     （x402 付款方的主要形态）
#
# 这个口径反而更正确 —— 它捕捉的是「决策频率」而非「交易发起方式」，
# 三组因此可比。论文中必须说明此口径及其理由。
ACTIVITY_KINDS = ["tx_from", "tx_to", "event"]


# ============================================================
# 🥈 备选数据源 —— Etherscan V2 多链统一接口
# ============================================================

ETHERSCAN_API_KEY = ""  # ← 留空则从环境变量 ETHERSCAN_API_KEY 读取

ETHERSCAN_V2_BASE = "https://api.etherscan.io/v2/api"

# 免费层限速：官方 5 req/s，实测建议留余量
RATE_LIMIT_PER_SEC = 4.0


# ============================================================
# 🔴 HTTP 限频总表 —— 2026-09-05 新增
# ============================================================
# 【为什么】 本项目对外有 7 类端点，此前 http_json（承担 Dune / agenteconomy /
#           GitHub raw 的全部调用）完全没有限频。Dune 免费层和 GitHub raw
#           （未认证 60 次/小时）都有明确限额，裸奔迟早撞 429。
#
# 【口径】 值 = 同一域名两次请求之间的最小间隔（秒）。
#         撞 429/5xx 时 DomainThrottle 会在此基线上追加惩罚间隔，
#         连续成功后再缓慢回落 —— 详见 src/utils/_base.py:DomainThrottle。
#
# 【依据】 各端点的公开限额与实测:
#   api.dune.com              免费层约 40 次/分 ⇒ 留一半余量取 1.5s
#   raw.githubusercontent.com 未认证 60 次/小时 ⇒ 本项目只在启动时取 1 次配置，
#                             1.0s 足够；真要频繁取应改用带 token 的 API
#   agenteconomy.to           静态站无公开限额，取 1.0s 作为礼貌间隔
#   _default                  未列出的域名兜底
HTTP_MIN_INTERVAL = {
    "api.dune.com": 1.5,
    "dune.com": 1.5,
    "raw.githubusercontent.com": 1.0,
    "agenteconomy.to": 1.0,
    "_default": 0.3,
}
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4

# 链 ID（Etherscan V2 chainid）
CHAINS = {
    "ethereum": 1,
    "base": 8453,
    "bsc": 56,
    "optimism": 10,
    "arbitrum": 42161,
    "polygon": 137,
    "gnosis": 100,
}

# 出块时间（秒）—— 决定延迟测量的分辨率下限
BLOCK_TIME = {
    "ethereum": 12.0,
    "base": 2.0,
    "bsc": 3.0,
    "optimism": 2.0,
    "arbitrum": 0.25,
    "polygon": 2.0,
    "gnosis": 5.0,
}

# ⚠️ 主战场选 Base：出块 2s，分辨率够区分 bot(<2s) 与 agent(2-10s)；
#    且 x402 / ERC-8004 / Clanker / Virtuals 主要活动都在 Base。
#    以太坊 12s 出块 → 分辨率不足以区分 bot 和 agent，只能做辅助验证。
PRIMARY_CHAIN = "base"


# ============================================================
# 采样与时间窗
# ============================================================
# 时间窗建议从 2025-09 起（x402 发布后），覆盖完整的暴涨-崩塌周期
START_BLOCK = 0          # 0 = 不限；若要限时间窗请填具体区块高度
END_BLOCK = 99999999

# Etherscan 单页记录数上限。
# ⚠️ 2026-07-01 起免费层从 10000 降到 1000 —— 填 >1000 会静默截断
ETHERSCAN_PAGE_SIZE = 1000
# 每个地址最多翻多少页（1000 × 20 = 2 万笔，足够本课题）
ETHERSCAN_MAX_PAGES = 20
MAX_TX_PER_ADDRESS = ETHERSCAN_PAGE_SIZE * ETHERSCAN_MAX_PAGES

# 每组最多采样多少地址（go/no-go 阶段不需要全量，几百个就能看出分布）
MAX_ADDR_PER_GROUP = 400

# 特征计算的最低交易数门槛 —— 低于此值的地址统计量不可靠，直接剔除
MIN_TX_FOR_FEATURES = 20


# ============================================================
# 🔴 按组自适应窗口倍数 —— 2026-09-03 实测修正
# ============================================================
# 【为什么需要】
#   12,000 块（≈6.6 小时）等长窗口实测结果:
#       bot   : 100 地址 / 83,537 条 = 835 条/地址
#       agent :  48 地址 / 12,234 条 = 255 条/地址
#       human : 100 地址 /    267 条 = 2.67 条/地址  ← 100% 低于门槛被剔除
#   相差 300 倍。这不是 BUG，是「人类天然低频」这一真实事实，
#   但它导致等长窗口下 human 组永远凑不满 MIN_TX_FOR_FEATURES。
#
# 【做法】 human 组窗口放大 12 倍（≈3.3 天），预期 2.67×12×0.92 ≈ 29 条/地址
#
# 【🔴 由此引入的新循环风险 —— 必须配套处理】
#   窗口不等长 ⇒ f_span_days / f_active_days / f_longest_silence_h
#   这三个「绝对跨度」特征会直接编码「你属于哪一组」，
#   模型只要学会「跨度 > 1 天 ⇒ human」就能作弊，与行为无关。
#   ⇒ 对策: 这三个特征在 f02_rhythm 中一律除以本组窗口长度做归一化，
#            并在 f_ensemble 中登记为 WINDOW_DEP_FEATURES 供消融对照。
# 🔴 2026-09-04 改为全组等长。此前 human ×20、另两组 ×1，
#    导致 entropy_dow / entropy_24h / active_hours 变成纯窗口伪影
#    （详见 f_ensemble.WINDOW_DEP_FEATURES 注释）。
#    等长后这些周期特征重新可比，是修根因而不是只做消融。
GROUP_WINDOW_MULT = {'agent': 1, 'bot': 1, 'human': 1}

# 周期类特征（星期熵/昼夜熵/活跃小时数）要有意义，窗口至少要覆盖一个完整周。
# Base 出块 2s ⇒ 7 天 = 7×86400/2 = 302,400 块。
# 这同时也满足了人类组的活动量需求（此前需 240,000 块）。
MIN_WINDOW_FOR_CYCLE_BLOCKS = 302_400

# 🔴 按组跳过的活动类型 —— 窗口 ×20 后每类扫描 606 次请求，必须精简
# 实测记录:
#   · human 组 240,000 块 tx_from 扫描 = 1,322s，命中 8,418 条（56 条/地址，已远超门槛 20）
#   · 同组 tx_to 扫描挂起 >12 小时未返回（SQD 侧 IO hang，非限流：全程仅 7 次 529）
#   ⇒ human 只扫 tx_from。人类的"决策时刻"本来就体现在自己发起的交易上，
#     被动收款（tx_to）反映的是别人的决策节奏，对 Δt 语义反而是噪声。
GROUP_SKIP_KINDS = {'human': {'tx_to', 'event'}}

# ============================================================
# 🔴 分段扫描 —— 2026-09-04 实测新增
# ============================================================
# 【问题】 bot 组在 7 天窗口下的 tx_from 数据量:
#            12,000 块 →  94,785 条 / 45 秒
#           302,400 块 → 约 240 万条 / 跑 100 分钟未完成
#         但每地址封顶 3,000 条，150 个地址最多留 45 万条 ——
#         拉回来的约 99% 当场被丢弃，纯粹浪费带宽和时间。
#
# 【做法】 只扫窗口内均匀分布的 12 个小段，而不是整段窗口。
#         这是「降密度、不降跨度」:
#           · 首末段仍相距 7 天 ⇒ entropy_dow / span_days 照常有效
#           · 12 段散布全周     ⇒ 覆盖不同星期与不同时段
#           · 段内区块连续      ⇒ 段内 Δt 是真实间隔
#           · 段间不算 Δt       ⇒ seg 字段 + f01 跨段过滤已兜底
#         等价于把 _merge_and_cap 的分段封顶从「先全拉再丢」
#         提前成「一开始只拉需要的」。
#
# 【谁开、谁不开】
#   bot   ✅ 数据量爆炸（240 万条），必须分段
#   agent ✅ 跟 bot 保持同一采样口径。实测全扫要 2 小时（其中 tx_to 独占 82
#            分钟只换来 170 条），分段后几分钟完成，且 event 从 487,757 条
#            降到够用的量级。两组同口径才能排除「采样方式差异」这一解释。
#   human ❌ 全扫。该组天然低频（整窗口仅 8,418 条），分段会直接把样本
#            打到 MIN_TX_FOR_FEATURES 门槛以下，整组被剔除。
#            ⚠️ 由此 human 与另两组采样密度不同 —— 所以密度类特征
#              (n_acts / acts_per_day) 必须留在 SCALE_FEATURES 里排除，
#              不得进模型。Δt 与周期类特征不受影响: 两种模式都保证
#              「段内区块连续」和「跨度覆盖完整窗口」。
GROUP_SEGMENT_SCAN = {'bot', 'agent'}

# 每段扫多少块。2,000 块 ≈ 66 分钟（Base 出块 2s），
# 对 bot 这种高频主体足够密集地采到连续行为。
SEGMENT_SCAN_BLOCKS = 2000


# ============================================================
# 🔴 30k 样本方案 —— 全窗口扫描 + 按活动数预筛（2026-09-07 试点标定定稿）
# ============================================================
# 【为什么重构采集】旧管线四层漏斗相乘只剩 247 个有效样本:
#   ① 发现窗口太小且在建模窗口之外（x402 只扫 20k 块、MEV 5k 块，均在窗口终点之后）
#   ② 每组只采样 150 个地址
#   ③ 采集窗口只有 7 天，agent/bot 还只扫其中 8%
#   ④ ≥20 条门槛把 agent/human 的通过率打到 35%/37%
# 【试点实测（2026-09-07，窗口内 3 个位置各 6,000 块）】
#   · x402 AuthorizationUsed: 10,685~15,677 条/6k块，去重付款方 867~1,990 个/6k块，
#     早期∩晚期重叠仅 3% —— 高流转 ⇒ 全窗口累计付款方在数十万级，agent 池充足
#   · DEX Swap: 59.4 条/块 ⇒ 全窗口扫描过重，bot 发现改为隔桶抽样（stride 3）
#   · 无过滤交易流: 706 笔/块 ⇒ 全窗口约 1.7 TB，「无过滤全量趟」不可行，
#     human 改为「探针预筛 + 地址过滤分批采集」
#   · logs + transactions 混合选择器一次查询: ✅ 可用（省一整趟全窗口行走）
# 【新架构】(sweep_window.py)
#   hotwallets → main(x402发现+事件活动+Olas exec+热钱包提币接收方，全窗口一趟)
#   → mev(隔桶) → probe(候选探针) → collect(地址过滤分批) → assemble(配额选样+落盘)
#   选样一律要求「全窗口活动数 ≥ MIN_TX_FOR_FEATURES」⇒ 入选即有效

SWEEP_DIR = DATA_DIR / "sweep"          # 扫描中间产物（分片/聚合/断点状态）

# 每组入选配额 = 30,000。
# 🔴 为什么不是三组等量 —— 成本感知的配额分配:
#   实测成本模型 = 地址数 × 扫描块数（SQD 服务端耗时随地址数线性增长:
#   1 选择器 1,000 地址 5.2s / 5 选择器 5,000 地址 26.2s，同样推进 505 块;
#   且单选择器硬上限 1,000 个，3,000 个直接 HTTP 400 ⇒ 无法靠批量摊薄）:
#     agent 事件  main 趟发现时顺带采完 ⇒ **零额外成本**
#     bot        36 段 × 10,000 块（4.5% 覆盖，高频足够）⇒ 约 1.4 万请求
#     human      必须全窗口连续（见 HUMAN_COVERAGE）⇒ 每 1,000 人 2 万请求
#   ⇒ 让零成本、池子最深的 agent 多担，把最贵的 human 压到 6,000。
# 【为什么 6,000 仍然够】 5 折 CV 下每折 1,200 个验证样本，指标已非常稳定
#   （原方案自评「每组 2,800 即可」）。不平衡比 2.2:1，用 class_weight +
#   macro-F1 处理，与既有口径完全一致。
GROUP_SELECT_QUOTA = {"agent": 13_000, "bot": 11_000, "human": 6_000}
TARGET_EFFECTIVE_TOTAL = 30_000
# 某组池子不足配额时，缺口按「成本从低到高」补给其他组: agent → bot → human
QUOTA_REBALANCE = True
QUOTA_REFILL_ORDER = ["agent", "bot", "human"]

# 选样过选比例。
# 🔴 为什么需要：assemble 的粗筛用「未去重计数」，写盘复核用「真实去重数」，
#    两个口径不一致 ⇒ 少量地址过了粗筛却卡在写盘门槛上，最终有效样本
#    少于 TARGET_EFFECTIVE_TOTAL（实测 30,000 目标下掉 4 个 agent → 29,996）。
#    对策：按此比例过选，写盘时按目标数精确截断，使最终有效样本恰好等于目标。
#    取 2% 是因为实测损耗率仅 0.013%，2% 已是两个数量级的余量。
SELECT_OVERSAMPLE_RATIO = 0.02

# 活动记录分段数（三组等长窗口、等分口径，与旧 N_SEGMENTS=12 同语义、粒度更细）
SWEEP_N_SEGMENTS = 24
# sweep 断点粒度：每 20,000 块一个 chunk，完成即落盘、重跑自动跳过
SWEEP_CHUNK_BLOCKS = 20_000
# 聚合快照间隔（每完成多少个 chunk 存一次全量聚合，供断点续跑）
SWEEP_SNAPSHOT_EVERY = 50

# ---- bot 发现（隔桶抽样，判据与原 fetch_mev_bots 同紧度）----
# 原判据是「5,000 块窗口内 arb_tx>=5」；全窗口若沿用绝对阈值会把 187 天里
# 偶发 5 次多跳 swap 的普通用户也收进来（试点实测 3,000 块就有 18,110 笔
# ≥2 swap 的交易，多数是聚合器路由）。改为「单个 5,000 块桶内 arb_tx>=5」
# —— 强度等价缩放，语义不变。
BOT_BUCKET_BLOCKS = 5_000
BOT_BUCKET_STRIDE = 3            # 每 3 桶扫 1 桶（覆盖 1/3 窗口，bot 有存续性）
BOT_MIN_SWAPS_IN_TX = 2
BOT_MIN_BUCKET_ARB = 5           # 单桶内原子套利笔数下限（对应原 min_arb=5）
BOT_MIN_BLOCKS = 3               # 距离原判据 min_blocks=3
# bot 采集分段：36 段 × 10,000 块（覆盖窗口 4.5%，高频主体足以采满）
BOT_COLLECT_SEGMENTS = 36
BOT_COLLECT_SEG_BLOCKS = 10_000

# ---- human 漏斗 ----
HOT_WALLET_SLICES = 6            # 热钱包发现：窗口内均布切片数
HOT_WALLET_SLICE_BLOCKS = 20_000
MAX_HOT_WALLETS_TOTAL = 60       # 各切片合并后最多保留几个热钱包
# human 漏斗是两级筛选，因为「全窗口采集」很贵，必须先把候选压小:
#   ① 免费预排序: main 趟已顺带统计每个接收方收到多少次 CEX 提币
#      (n_from_cex)。收提币越频繁 ⇒ 越可能是在用的活钱包。
#      这一步零成本，先把几十万接收方压到 15,000。
#   ② 探针精排: 对这 15,000 个扫窗口的 5%，按自发交易命中数排名，
#      取前 7,500 进全窗口采集。
# 🔴 探针只看「用了多少次」这一个标量，绝不看时间分布 ——
#   任何涉及时段的筛选都会让 entropy_24h / night_ratio 变成「用答案筛样本」。
HUMAN_PRERANK_BY_WITHDRAWALS = True   # ① 用 n_from_cex 免费预排序
HUMAN_PROBE_POOL = 15_000        # 进探针的候选数上限（每批 1,000 个地址）
HUMAN_PROBE_SEGMENTS = 24        # 探针段数（均布，比连续段更抗突发）
HUMAN_PROBE_SEG_BLOCKS = 16_800  # 每段块数 ⇒ 覆盖率 24×16.8k/8.06M ≈ 5%
HUMAN_PROBE_KEEP = 7_500         # 探针排名进入全窗口采集的人数（quota×1.25）
# 🔴 human 正式采集必须是「全窗口连续」（覆盖率 1.0），不能分段。
# 【为什么】 f01_latency 只在同一 seg 内算 Δt（跨段是采样造成的人为断点，
#   不是真实间隔）。人类天然低频 —— 全窗口 40 笔 ≈ 0.21 笔/天。
#   若按 24 个连续大段（每段 3.9 天）采集，每段期望仅 0.8 笔 ⇒ 段内几乎
#   凑不出相邻对 ⇒ Δt 有效样本 <5 ⇒ **整组在特征阶段被剔除**。
#   （这正是旧管线 human 组只剩 56 个的根因之一。）
#   反之全窗口连续采集时 seg 恒为 0，N 笔活动给出 N-1 个真实 Δt。
# 【代价】 这是本方案最贵的一步（每 1,000 人约 2 万请求）——
#   所以用探针先把候选压到 11,000，而不是盲采几万个。
# 【bot/agent 为什么可以分段】 它们高频，单段内就有几十上百笔，段内 Δt 充足。
HUMAN_COVERAGE = 1.0
HUMAN_MAX_ACTS_WINDOW = 2_000    # 全窗口自发交易上限（≈10.7 笔/天，防混入 bot）

# ---- x402 提纯 ----
# EIP-3009 不止 x402 在用（Coinbase 智能钱包的免 gas 转账同样触发
# AuthorizationUsed）。用「提交者画像」区分：x402 facilitator 结算的是
# 海量小额（官方口径均单 $0.31），零售中继提交的是大额转账。
# 画像要拉全部 USDC Transfer 来配对金额，而 Base 上 USDC 转账量极大 ——
# 实测 6,000 块/切片时单切片要跑十几分钟。分类几百个提交者并不需要那么多样本
# （3,000 块即有数千条 auth 事件），故收窄到 3,000 块 × 6 切片。
X402_PROFILE_SLICES = 6          # facilitator 画像采样切片数（配对金额用）
X402_PROFILE_SLICE_BLOCKS = 3_000
X402_FACIL_GOOD_MAX_AVG = 5.0    # 均单 ≤$5 的提交者视为 x402 型
X402_FACIL_BAD_MIN_AVG = 20.0    # 均单 ≥$20 的提交者视为零售型
X402_BAD_SUBMITTER_MAX_SHARE = 0.30   # 经零售型提交者的支付占比超此值剔除该付款方
X402_MIN_PAYMENTS_CSV = 20       # 写入 x402_payers.csv 的最低支付数（=有效门槛）

# ---- Olas（Base 高置信层）----
OLAS_COLLECT_FULL_WINDOW = True  # 是否为 Olas Safe 单独跑全窗口 tx_from/tx_to
SAFE_EXEC_TOPIC = True           # main 趟顺带采 Safe ExecutionSuccess 事件

# ============================================================
# 假设的三个延迟频带（秒）—— 对应 H1
# ============================================================
BAND_BOT = (0.0, 2.0)        # 亚秒 ~ 2 秒
BAND_AGENT = (2.0, 10.0)     # ★ LLM 推理延迟带
BAND_HUMAN = (10.0, 1e9)     # >10 秒

BAND_LABELS = ["<2s (bot带)", "2-10s (agent带)", ">10s (human带)"]

# 🔴 实测发现的边界冲突（用 Base 系统地址验证）
# ------------------------------------------------------------
# Base 出块 2s，意味着 Δt 在物理上只能取 {0, 2, 4, 6, ...}：
#     Δt = 0s  同区块多笔        → 确定是 bot
#     Δt = 2s  连续区块          → ⚠️ bot 和「快 agent」不可分，落在 agent 带边界上
#     Δt ≥ 4s  隔 2 个区块以上   → 才是干净的 agent 带
#
# 实测：Base 系统地址（每区块必发 1 笔，是纯粹的 bot）算出
#       frac_2to10s = 1.0、dt_p50 = 2.0 —— 100% 落进「agent 带」，是假阳性。
#
# 对策（论文里必须交代）：
#   1. 分析时把 Δt = 0 和 Δt = 一个出块间隔 单独成桶，不要混进 agent 带
#   2. same_block_ratio 与 consec_block_ratio 在亚区块尺度上比 Δt 更有判别力
#   3. 若要真正分辨 <2s 的差异，必须引入 mempool 数据（交易何时被广播，
#      而非何时被打包）—— 这是完整版论文的必要工作
BLOCK_TIME_FLOOR_WARNING = True


# ============================================================
# go/no-go 判定阈值
# ============================================================
# 见 src/plot_gonogo.py。三条线同时看：
#   1) 三分类 macro-F1
#   2) agent 类召回率
#   3) 二分类对照实验的 agent 漏检率（对标 Web 域的 34.5%-39.1%）
GONOGO_F1_GO = 0.70
GONOGO_F1_CAUTION = 0.55
GONOGO_AGENT_RECALL_GO = 0.60
GONOGO_AGENT_RECALL_MIN = 0.40

RANDOM_SEED = 42

# ============================================================
# Coinglass付费服务
# ============================================================
# 🔴 本常量必须留空 —— 本文件未被 .gitignore 忽略，填在这里的 key 会进版本库。
#    2026-09-13 审计：此处曾硬编码过一枚明文 key，已清空。
#    ⚠️ 曾经提交过的 key 视同已泄露，必须到 CoinGlass 控制台吊销重签，
#       光删掉本行不够（历史提交里仍可检出）。
#    正确用法：
#        export COINGLASS_API_KEY=你的key
#    代码读取顺序恒为：构造参数 > 环境变量 COINGLASS_API_KEY > 本常量
#    key 为空时 run_all.sh 会自动跳过 Coinglass 阶段（它不参与建模，不影响主结论）。

COIN_GLASS_KEY = ''

# ============================================================
# 三组样本的定义
# ============================================================

GROUPS = {
    "agent": ("agent_olas.csv", "AI Agent (Olas/x402)", "#2f6fb0"),
    "bot":   ("bot_mev.csv",    "传统脚本 Bot (MEV)",    "#b03636"),
    "human": ("human_cex.csv",  "人类 (CEX 提现)",       "#2b7a4b"),
}


def get_api_key() -> str:
    """优先环境变量，其次本文件常量。"""
    import os
    key = os.environ.get("ETHERSCAN_API_KEY", "").strip() or ETHERSCAN_API_KEY.strip()
    if not key:
        raise SystemExit(
            "\n[配置错误] 缺少 Etherscan API key。\n"
            "  1) 免费申请：https://etherscan.io/apis\n"
            "  2) 然后执行：export ETHERSCAN_API_KEY=你的key\n"
            "     或直接填入 config.py 的 ETHERSCAN_API_KEY\n"
            "\n  想先跑通管线？用合成数据，不需要 key：\n"
            "     python src/make_synthetic.py && python src/compute_features.py --synthetic\n"
        )
    return key
