"""
AlphaGPT-Routine **v2 (A股优化版)**

在 kafroc 版基础上，结合原作者 imbue-bit/AlphaGPT 的核心原理重写。

================================================================================
【一、保留的「原始版核心原理」——这才是 AlphaGPT 的灵魂】
================================================================================
1. 闭环自进化：Transformer 生成公式 → StackVM 解释执行 → 回测打分 → REINFORCE 更新生成器。
   不是预测价格，而是"自动写因子"。
2. **奖励函数 = 你对好策略的定义**（原版最值钱的设计）：
   - 真实交易成本（手续费 + 冲击成本）而不是理论零成本
   - 交易次数下限（操作太少的公式直接淘汰，防偶然）
   - 大回撤二次惩罚（原版：(单笔亏损>5%) 的次数 × 2）
   - 风险调整收益做主轴（原版用 Sortino，而不是胜率/绝对收益）
   - 公式长度惩罚（原版注释："短小精悍的公式往往更稳"）
3. 严格样本外：前 80% 只用于搜索，后 20% 只做一次 reality check。
4. 可解释 + 语法合法：波兰式 token 序列 + 严格掩码保证公式是合法表达式树。

================================================================================
【二、修复 v1 的致命问题】
================================================================================
[致命] F_BUY_F_REPLAY 未归一化：量级 1e8，而其他 5 个因子被 robust_norm 压到 ±5。
       任何含该因子的公式都被它支配，tanh 饱和 → 信号退化成常数。
       修复：改成「融资净买入 / 融资余额」的相对量，再做 robust_norm。

[致命] 奖励函数被改成「胜率最大化」（win_rate_pct）。
       胜率是可以靠"赚小钱、扛大亏"刷高的危险目标，会选出爆仓型策略。
       修复：恢复原版的「风险调整收益」思路，用 Sortino 为主轴。

[致命] 收益标签用「未来 2~11 天里第一个正收益的开盘价」卖出，
       于是每天一个信号、持有 11 天，**11 个重叠的多日收益被当成 11 个独立日收益**做复利，
       年化和夏普被系统性放大好几倍。
       修复：改成「持仓状态机」——持仓期间不开新仓，得到真正不重叠的日频净值序列。

================================================================================
【三、A股制度适配】
================================================================================
- T+1：买入次日才能卖出（退出候选从买入日之后开始）
- 涨跌停：开盘涨停买不进 / 开盘跌停卖不出，自动顺延到下一个可成交日
- 停牌/零成交：跳过，不参与交易
- 真实成本：佣金 + 过户费 + 印花税（仅卖出）+ 滑点，买卖两侧分开计
- 基准对比：同期买入持有（Buy & Hold）
"""

import base64
import glob
import hashlib
import hmac
import json
import os
import time
import urllib.parse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm


# ==============================================================================
# 0. 配置（全部可用环境变量覆盖）
# ==============================================================================
def _get_env(key, default, cast_type=str):
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    if cast_type == bool:
        return str(val).lower() in ("true", "yes", "1", "y")
    return cast_type(val)


# ---- 标的池与区间 ----
# 【原始版核心原理之一】在「横截面」上训练，而不是单只股票。
# 原版 imbue 用 500 个代币同时训练；单只股票只有约 900 个交易日，
# 在这个样本量上做强化学习搜索必然过拟合。
# 默认标的池：沪市流动性好的两融标的（两融缓存只有沪市数据，深市标的该因子会为 0）
_DEFAULT_UNIVERSE = (
    "600519,600036,601318,600030,600276,600887,"
    "601899,600900,601088,600309,601166,600585"
)
_UNIVERSE_ENV = os.environ.get("UNIVERSE", "").strip()
_INDEX_CODE_ENV = os.environ.get("INDEX_CODE", "").strip()
if _UNIVERSE_ENV:                         # UNIVERSE 优先（显式指定标的池）
    UNIVERSE = [c.strip() for c in _UNIVERSE_ENV.split(",") if c.strip()]
elif _INDEX_CODE_ENV:                     # 兼容 v1 的单标的用法
    UNIVERSE = [c.strip() for c in _INDEX_CODE_ENV.split(",") if c.strip()]
else:
    UNIVERSE = [c.strip() for c in _DEFAULT_UNIVERSE.split(",") if c.strip()]
INDEX_CODE = UNIVERSE[0]                  # 用于文件命名与推送标题
STOCK_NAMES = {
    "600519": "贵州茅台", "600036": "招商银行", "601318": "中国平安", "600030": "中信证券",
    "600276": "恒瑞医药", "600887": "伊利股份", "601899": "紫金矿业", "600900": "长江电力",
    "601088": "中国神华", "600309": "万华化学", "601166": "兴业银行", "600585": "海螺水泥",
}

START_DATE = _get_env("START_DATE", "20220101")
END_DATE = _get_env("END_DATE", "20261231")

# ---- 模型 / 搜索 ----
BATCH_SIZE = _get_env("BATCH_SIZE", 1024, int)
TRAIN_ITERATIONS = _get_env("TRAIN_ITERATIONS", 100, int)
MAX_SEQ_LEN = _get_env("MAX_SEQ_LEN", 10, int)
FORCE_TRAIN = _get_env("FORCE_TRAIN", True, bool)
ONLY_LONG = _get_env("ONLY_LONG", True, bool)
SEED = _get_env("SEED", 42, int)

# ---- 交易规则（A股）----
HOLD_PERIOD = _get_env("HOLD_PERIOD", 5, int)      # 固定持有交易日数（v1 是 11 天的择时，已改为固定持有）
COMMISSION_RATE = _get_env("COMMISSION_RATE", 0.00025, float)   # 佣金，双边
TRANSFER_FEE_RATE = _get_env("TRANSFER_FEE_RATE", 0.00001, float)  # 过户费，双边
STAMP_TAX_RATE = _get_env("STAMP_TAX_RATE", 0.0005, float)      # 印花税，仅卖出
SLIPPAGE_RATE = _get_env("SLIPPAGE_RATE", 0.0005, float)        # 滑点，单边
LIMIT_RATE_OVERRIDE = _get_env("LIMIT_RATE", 0.0, float)        # >0 则强制涨跌停比例（ST股设 0.05）

BUY_COST = COMMISSION_RATE + TRANSFER_FEE_RATE + SLIPPAGE_RATE
SELL_COST = COMMISSION_RATE + TRANSFER_FEE_RATE + STAMP_TAX_RATE + SLIPPAGE_RATE

# ---- 奖励函数（原始版核心）----
MIN_TRADES = _get_env("MIN_TRADES", 12, int)        # 【单只】交易次数下限，不足则淘汰该标的
MIN_STOCK_PASS = _get_env("MIN_STOCK_PASS", 0.9, float)  # 至少多少比例的标的达标，公式才算有效
TOP_N = _get_env("TOP_N", 5, int)                    # 【核心】组合持有股票数（横截面排序取前 N）
MIN_PICKS = _get_env("MIN_PICKS", 3, int)            # 合格标的少于这个数就空仓（防止 100% 押一只）
REQUIRE_POSITIVE = _get_env("REQUIRE_POSITIVE", True, bool)  # 只买 signal>0 的股票
MIN_PORTFOLIO_TRADES = _get_env("MIN_PORTFOLIO_TRADES", 30, int)  # 组合交易次数下限
BATCH_CHUNK = _get_env("BATCH_CHUNK", 64, int)       # 回测分块大小（控制内存）
MAX_DD_PENALTY = _get_env("MAX_DD_PENALTY", 3.0, float)   # 回撤超阈值后的惩罚系数
DD_THRESHOLD = _get_env("DD_THRESHOLD", 0.25, float)      # 回撤惩罚阈值
LEN_PENALTY = _get_env("LEN_PENALTY", 0.03, float)        # 每个 token 的长度惩罚
RISK_FREE_RATE = _get_env("RISK_FREE_RATE", 0.02, float)  # 无风险利率（算 Sortino 用）

LAST_NDAYS = _get_env("LAST_NDAYS", 10, int)
BEST_FORMULA = _get_env("BEST_FORMULA", "")
DINGTALK_WEBHOOK = _get_env("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET = _get_env("DINGTALK_SECRET", "")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# 固定随机种子：让"同一个配置跑两次"结果一致（v1 每次结果都不同，无法复现）
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# 1. 钉钉推送（沿用 v1，格式升级）
# ==============================================================================
def send_dingtalk_msg(text):
    if not DINGTALK_WEBHOOK:
        return False
    url = DINGTALK_WEBHOOK
    if DINGTALK_SECRET:
        timestamp = str(round(time.time() * 1000))
        string_to_sign = "{}\n{}".format(timestamp, DINGTALK_SECRET)
        hmac_code = hmac.new(
            DINGTALK_SECRET.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote(base64.b64encode(hmac_code))
        url = f"{DINGTALK_WEBHOOK}&timestamp={timestamp}&sign={sign}"
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": "AlphaGPT Strategy Notification", "text": text},
    }
    try:
        resp = requests.post(
            url, headers={"Content-Type": "application/json"},
            data=json.dumps(payload), timeout=10,
        )
        body = resp.json()
        print(f"DingTalk notification sent, status: {resp.status_code}, errcode={body.get('errcode')}")
        return body.get("errcode") == 0
    except Exception as e:
        print(f"Failed to send DingTalk notification: {e}")
        return False


# ==============================================================================
# 2. 因子语言：算子 + 特征 + 词表
# ==============================================================================
@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0:
        return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)


@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y


@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)


@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)


OPS_CONFIG = [
    ("ADD", lambda x, y: x + y, 2),
    ("SUB", lambda x, y: x - y, 2),
    ("MUL", lambda x, y: x * y, 2),
    ("DIV", lambda x, y: x / (y + 1e-6), 2),
    ("NEG", lambda x: -x, 1),
    ("ABS", torch.abs, 1),
    ("SIGN", torch.sign, 1),
    ("GATE", _op_gate, 3),
    ("JUMP", _op_jump, 1),
    ("DECAY", _op_decay, 1),
    ("DELAY1", lambda x: _ts_delay(x, 1), 1),
    ("MAX3", lambda x: torch.max(x, torch.max(_ts_delay(x, 1), _ts_delay(x, 2))), 1),
]

# 全部可选因子（固定顺序，"MARGIN_NET" 由两融矩阵单独填充）
_ALL_FACTORS = ["RET", "RET5", "VOL_CHG", "V_RET", "TREND", "MARGIN_NET"]
# 实际使用的因子集：可用环境变量控制，逗号分隔。
# 例：FACTORS=RET,RET5,VOL_CHG,V_RET,TREND  → 只用价量因子（全市场数据完整）
_FACTORS_ENV = _get_env("FACTORS", "")
FEATURES = [f.strip().upper() for f in (_FACTORS_ENV.split(",") if _FACTORS_ENV else _ALL_FACTORS)
            if f.strip()]
_bad = [f for f in FEATURES if f not in _ALL_FACTORS]
if _bad:
    raise ValueError(f"未知因子 {_bad}，可选: {_ALL_FACTORS}")
if not FEATURES:
    raise ValueError("至少选择一个因子")

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
VOCAB_SIZE = len(VOCAB)
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}


# ==============================================================================
# 3. 模型（与原始版 times.py 逐字一致，只保留 PyTorch 兼容性修复）
# ==============================================================================
class AlphaGPT(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=128,
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head_actor = nn.Linear(d_model, VOCAB_SIZE)
        self.head_critic = nn.Linear(d_model, 1)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        return self.head_actor(last), self.head_critic(last)


# ==============================================================================
# 4. A股交易制度工具
# ==============================================================================
def limit_rate_for(code):
    """按板块返回涨跌停比例。ST 股无法从代码判断，用 LIMIT_RATE 环境变量强制。"""
    if LIMIT_RATE_OVERRIDE > 0:
        return LIMIT_RATE_OVERRIDE
    c = str(code)
    if c.startswith(("300", "301", "688")):
        return 0.20      # 创业板 / 科创板
    if c.startswith(("4", "8", "9")):
        return 0.30      # 北交所
    return 0.10          # 沪深主板


def build_tradability(open_, close_, volume, code):
    """
    生成「能不能买 / 能不能卖」掩码。

    - 停牌、零成交：不可交易
    - 开盘涨停：买不进（我们假设以开盘价成交，涨停开盘即无法建仓）
    - 开盘跌停：卖不出（顺延）

    返回 (entry_ok, exit_ok, halt)
    """
    n = len(open_)
    rate = limit_rate_for(code)
    prev_close = np.concatenate([[close_[0]], close_[:-1]])
    halt = (volume <= 0) | (open_ <= 0) | (close_ <= 0)
    open_limit_up = open_ >= prev_close * (1 + rate) * (1 - 0.005)
    open_limit_down = open_ <= prev_close * (1 - rate) * (1 + 0.005)
    entry_ok = ~halt & ~open_limit_up
    exit_ok = ~halt & ~open_limit_down
    return entry_ok, exit_ok, halt


def simulate_positions(signal, open_, close_, entry_ok, exit_ok, hold_period):
    """
    持仓状态机：把「每日信号」变成「不重叠的真实持仓」。

    规则（A股可执行口径）：
      - signal[t] == 1 → t+1 开盘买入（若 t+1 涨停/停牌则放弃该信号）
      - 持有 hold_period 个交易日后，在该日开盘卖出（T+1 已隐含：hold_period>=1）
      - 卖出日若跌停/停牌 → 顺延到下一个可成交日
      - **持仓期间不再开新仓**（这是与 v1 最大的区别：保证收益不重叠）

    返回
      daily_ret : 每日净值收益（含买卖成本），长度 n
      trades    : 每笔交易的记录
      pos_flags : 每日是否持仓
    """
    n = len(signal)
    daily_ret = np.zeros(n)
    pos_flags = np.zeros(n)
    trades = []
    i = 0
    while i < n - 1:
        if signal[i] > 0:
            e = i + 1
            if not entry_ok[e]:
                i += 1
                continue
            k = e + max(1, hold_period)          # T+1 起才可卖
            while k < n and not exit_ok[k]:      # 跌停/停牌 → 顺延
                k += 1
            if k >= n:
                # 数据结束仍未平仓：按市值计入净值（真实持有），但不计为一笔已完成交易
                pos_flags[e:n] = 1
                daily_ret[e] += close_[e] / open_[e] - 1.0 - BUY_COST
                for t in range(e + 1, n):
                    daily_ret[t] += close_[t] / close_[t - 1] - 1.0
                break
            entry_px = open_[e]
            exit_px = open_[k]
            if entry_px <= 0 or exit_px <= 0:
                i += 1
                continue

            pos_flags[e:k] = 1
            # ---- 本笔交易贡献的日收益（入场日按开盘买、卖出日按开盘卖，中间按收盘）----
            seg = [close_[e] / entry_px - 1.0 - BUY_COST]          # 入场日（扣买入成本）
            seg += [close_[t] / close_[t - 1] - 1.0 for t in range(e + 1, k)]
            seg.append(exit_px / close_[k - 1] - 1.0 - SELL_COST)  # 卖出日（扣卖出成本）
            for t, r in zip(range(e, k + 1), seg):
                daily_ret[t] += r
            # 净收益 = 本笔日收益复利，保证与净值曲线严格一致
            net = float(np.prod([1.0 + r for r in seg]) - 1.0)

            gross = exit_px / entry_px - 1.0
            trades.append({
                "entry_date": e, "exit_date": k,
                "entry_px": float(entry_px), "exit_px": float(exit_px),
                "gross": float(gross), "net": float(net),
                "days": int(k - e), "forced": bool(k > e + max(1, hold_period)),
            })
            i = k                                # 平仓后才考虑下一个信号
        else:
            i += 1
    return daily_ret, trades, pos_flags


def simulate_batch(signal, open_, close_, entry_ok, exit_ok, hold_period):
    """
    simulate_positions 的向量化版本：一次并行模拟 S 条序列（S = 样本数 × 标的数）。

    与 simulate_positions 规则完全一致：
      昨天信号 → 今天开盘买；持有 hold_period 个交易日后开盘卖；跌停/停牌顺延；
      持仓期间不开新仓（不重叠）。

    signal/open_/close_/entry_ok/exit_ok 形状均为 [S, T]
    返回 (daily_ret [S,T], pos_flags [S,T], stats dict)
    """
    S, T = signal.shape
    hold = max(1, int(hold_period))
    daily = np.zeros((S, T), dtype=np.float64)
    flag = np.zeros((S, T), dtype=bool)
    flat = np.ones(S, dtype=bool)
    entry_px = np.zeros(S)
    exit_day = np.zeros(S, dtype=np.int64)
    seg = np.ones(S)
    n_tr = np.zeros(S, dtype=np.int64)
    wins = np.zeros(S, dtype=np.int64)
    forced = np.zeros(S, dtype=np.int64)
    sum_net = np.zeros(S)
    best = np.full(S, -np.inf)
    worst = np.full(S, np.inf)

    for t in range(1, T):
        was_holding = ~flat
        new_entry = flat & (signal[:, t - 1] > 0) & entry_ok[:, t]
        exiting = was_holding & (t >= exit_day) & exit_ok[:, t]
        cont = was_holding & (~exiting)

        day_ret = np.zeros(S)
        if cont.any():
            day_ret[cont] = close_[cont, t] / close_[cont, t - 1] - 1.0
        e = np.where(new_entry)[0]
        if e.size:
            day_ret[e] = close_[e, t] / open_[e, t] - 1.0 - BUY_COST
        x = np.where(exiting)[0]
        if x.size:
            day_ret[x] = open_[x, t] / close_[x, t - 1] - 1.0 - SELL_COST

        if e.size:                                   # 建仓
            flat[e] = False
            entry_px[e] = open_[e, t]
            exit_day[e] = t + hold
            seg[e] = 1.0
        if x.size:                                   # 平仓结算
            net = seg[x] * (1.0 + day_ret[x]) - 1.0
            n_tr[x] += 1
            wins[x] += (net > 0)
            forced[x] += (t > exit_day[x])
            sum_net[x] += net
            best[x] = np.maximum(best[x], net)
            worst[x] = np.minimum(worst[x], net)
            flat[x] = True

        active = ~flat                               # 收盘仍持仓
        if active.any():
            seg[active] *= (1.0 + day_ret[active])
        daily[:, t] = day_ret
        flag[active, t] = True

    empty = n_tr == 0
    best[empty] = 0.0
    worst[empty] = 0.0
    return daily, flag, {
        "n_trades": n_tr, "wins": wins, "forced": forced,
        "sum_net": sum_net, "best": best, "worst": worst,
        "avg_net": np.where(n_tr > 0, sum_net / np.maximum(n_tr, 1), 0.0),
    }


def simulate_topn_batch(factors, open_, close_, entry_ok, exit_ok, hold, n_top,
                        require_positive=True):
    """
    【核心】横截面 Top-N 组合模拟 —— 对应原始版实盘 runner「按分数排序取前 N 只」的做法。

    与 v1/旧 v2 的区别：
      旧：每只股票各自出二值信号 → 30 只里常常 14 只同时买入 ≈ 半个指数（被稀释）
      新：每天只在全池里挑因子值最高的 N 只持有，资金等权 → 真正的组合

    规则：
      - 空仓时每天看一次：取 signal>0 且当日可买 的股票中因子值前 N 只
      - t 日收盘决策 → t+1 开盘等权买入（每只权重 1/N）
      - 持有 hold 个交易日后卖出；跌停/停牌则顺延到能卖的那天
      - 持仓期间不再调仓（等这一批全部平掉再重新选股）

    factors: [B, N_stock, T] 原始因子值
    返回 (daily_ret [B, T], stats dict)
    """
    B, N, T = factors.shape
    hold = max(1, int(hold))
    n_top = max(int(n_top), int(MIN_PICKS))
    sig = np.tanh(factors)
    daily = np.zeros((B, T), dtype=np.float64)
    active = np.zeros((B, N), dtype=bool)
    pos_cum = np.ones((B, N), dtype=np.float64)
    m = np.zeros(B, dtype=np.int64)              # 本批持仓数
    exit_target = np.zeros(B, dtype=np.int64)
    holding = np.zeros(B, dtype=bool)
    n_trades = np.zeros(B, dtype=np.int64)
    wins = np.zeros(B, dtype=np.int64)
    forced = np.zeros(B, dtype=np.int64)
    sum_net = np.zeros(B)
    day_count = np.zeros(B, dtype=np.int64)
    exp_day = np.zeros(B, dtype=np.int64)
    pick_count = np.zeros((B, N), dtype=np.int64)
    cohort = np.zeros((B, N), dtype=bool)      # 本批次的成员（用于按市值加总）
    rows_all = np.arange(B)

    for t in range(1, T):
        entered = np.zeros((B, N), dtype=bool)
        # 【关键】批次内按「买入持有」算组合市值：不做每日再平衡
        # v = Σ(每只当前市值) / 批次持仓数；日收益 = v_t / v_{t-1} - 1
        v_prev = np.where(holding, (pos_cum * cohort).sum(axis=1) / np.maximum(m, 1), 1.0)
        o_t = open_[:, :, t]            # [B, N] 当日开盘
        c_t = close_[:, :, t]           # [B, N] 当日收盘
        c_p = close_[:, :, t - 1]       # 前一日收盘
        e_ok = entry_ok[:, :, t]
        x_ok = exit_ok[:, :, t]

        # ---------- 1) 空仓的样本尝试建仓（用 t-1 收盘的因子） ----------
        idle = ~holding
        if idle.any():
            prev_sig = sig[:, :, t - 1]
            elig = idle[:, None] & e_ok
            if require_positive:
                elig = elig & (prev_sig > 0)
            score = np.where(elig, prev_sig, -np.inf)
            order = np.argsort(-score, axis=1)[:, :n_top]
            picked = np.zeros((B, N), dtype=bool)
            picked[rows_all[:, None], order] = True
            picked &= elig
            cnt = picked.sum(axis=1)
            # 【风控】合格标的不足 MIN_PICKS 就空仓，避免把全部资金压在一两只上
            take = idle & (cnt >= MIN_PICKS)
            if take.any():
                active[take] = picked[take]
                cohort[take] = picked[take]
                pos_cum[take] = 1.0
                m[take] = cnt[take]
                exit_target[take] = t + hold
                holding[take] = True
                entered[take] = active[take]
                pick_count[take] += picked[take]

        # ---------- 2) 当日收益 ----------
        day_ret = np.zeros((B, N))
        will_exit = active & (t >= exit_target[:, None]) & x_ok     # 今天开盘卖
        if will_exit.any():
            day_ret[will_exit] = o_t[will_exit] / c_p[will_exit] - 1.0 - SELL_COST
            forced += (will_exit & (t > exit_target[:, None])).sum(axis=1)   # 因跌停/停牌顺延
        if entered.any():                                           # 今天开盘买
            day_ret[entered] = c_t[entered] / o_t[entered] - 1.0 - BUY_COST
        cont = active & (~will_exit) & (~entered)                   # 继续持有
        if cont.any():
            day_ret[cont] = c_t[cont] / c_p[cont] - 1.0

        pos_cum = np.where(active, pos_cum * (1.0 + day_ret), pos_cum)
        v_new = (pos_cum * cohort).sum(axis=1) / np.maximum(m, 1)   # 建仓日 m 已更新，必须重算
        traded = holding | entered.any(axis=1)
        daily[:, t] = np.where(traded, v_new / np.where(v_prev == 0, 1.0, v_prev) - 1.0, 0.0)

        # ---------- 3) 结算卖出的仓位 ----------
        if will_exit.any():
            net = pos_cum - 1.0
            n_trades += will_exit.sum(axis=1)
            wins += ((net > 0) & will_exit).sum(axis=1)
            sum_net += np.where(will_exit, net, 0.0).sum(axis=1)
            active[will_exit] = False
            pos_cum[will_exit] = 1.0

        # ---------- 4) 批次结束判定 ----------
        still = active.any(axis=1)
        exp_day[still] += 1
        day_count += 1
        finished = holding & (~still)
        holding[finished] = False
        m[finished] = 0
        cohort[finished] = False

    return daily, {
        "n_trades": n_trades, "wins": wins, "sum_net": sum_net,
        "avg_net": np.where(n_trades > 0, sum_net / np.maximum(n_trades, 1), 0.0),
        "exposure": exp_day / np.maximum(day_count, 1),
        "forced": forced,
        "pick_count": pick_count,
    }


def perf_stats(daily_ret):
    """由日频收益序列算真实绩效（年化按 252 交易日）。"""
    n = len(daily_ret)
    if n == 0:
        return {}
    if np.allclose(daily_ret, 0.0):
        return {"total": 0.0, "ann": 0.0, "vol": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_dd": 0.0, "calmar": 0.0, "equity": np.ones(n)}
    equity = np.cumprod(1.0 + daily_ret)
    total = equity[-1] - 1.0
    years = n / 252.0
    ann = equity[-1] ** (1.0 / years) - 1.0 if equity[-1] > 0 else -1.0
    vol = daily_ret.std() * np.sqrt(252)
    downside = daily_ret[daily_ret < 0]
    dvol = downside.std() * np.sqrt(252) if downside.size > 1 else 0.0
    sharpe = (ann - RISK_FREE_RATE) / (vol + 1e-9)
    sortino = (ann - RISK_FREE_RATE) / (dvol + 1e-9)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(1.0 - equity / peak)) if n else 0.0
    return {
        "total": total, "ann": ann, "vol": vol, "sharpe": sharpe,
        "sortino": sortino, "max_dd": max_dd,
        "calmar": ann / (max_dd + 1e-9),
        "equity": equity,
    }


# ==============================================================================
# 5. 数据引擎（横截面版：一次加载整个标的池）
# ==============================================================================
def robust_norm(x):
    """中位数/MAD 标准化 + 截断到 ±5。所有因子都必须走这一步（v1 第 6 个因子漏了，量级 1e8）。"""
    x = np.asarray(x, dtype=np.float32)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median)) + 1e-6
    return np.clip((x - median) / mad, -5, 5).astype(np.float32)


_BS_LOGGED_IN = False


def _bs_login_once():
    global _BS_LOGGED_IN
    import baostock as bs
    if not _BS_LOGGED_IN:
        bs.login()
        _BS_LOGGED_IN = True
    return bs


def fetch_daily(code):
    """取单只股票日线，返回 (DataFrame[date,open,high,low,close,volume], 数据源)。"""
    try:
        import akshare as ak
        raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START_DATE,
                                 end_date=END_DATE, adjust="qfq")
        if raw is None or raw.empty:
            raise ValueError("akshare 返回空")
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["日期"]),
            "open": pd.to_numeric(raw["开盘"], errors="coerce"),
            "high": pd.to_numeric(raw["最高"], errors="coerce"),
            "low": pd.to_numeric(raw["最低"], errors="coerce"),
            "close": pd.to_numeric(raw["收盘"], errors="coerce"),
            "volume": pd.to_numeric(raw["成交量"], errors="coerce"),
        })
        return df.sort_values("date").reset_index(drop=True), "akshare"
    except Exception:
        bs = _bs_login_once()
        prefix = "sh." if code.startswith(("6", "9")) else "sz."
        rs = bs.query_history_k_data_plus(
            prefix + code, "date,open,high,low,close,volume",
            start_date=pd.to_datetime(START_DATE).strftime("%Y-%m-%d"),
            end_date=pd.to_datetime(END_DATE).strftime("%Y-%m-%d"),
            frequency="d", adjustflag="2")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            raise ValueError(f"{code} 未取到数据")
        df = pd.DataFrame(rows, columns=rs.fields)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True), "baostock"


def load_margin_balance_matrix(codes, date_strs, cache_file="margin_balance_matrix.parquet"):
    """
    构建 [日期 × 标的] 的「融资余额」矩阵，数据来源：
      - 沪市：仓库自带的 margin_balance/YYYYMMDD_margin_data.parquet（按日，含「融资余额」）
      - 深市：fetch_margin_szse.py 抓取的 margin_szse_cache.parquet

    注意：深交所明细里**没有「融资偿还额」**字段，沪市有。
    所以 v2 统一改用「融资余额的变化率」作为因子 —— 两市都能算，口径一致：
        MARGIN_NET = (余额[t] - 余额[t-1]) / |余额[t-1]|
    """
    if os.path.exists(cache_file):
        try:
            mat = pd.read_parquet(cache_file)
            if set(date_strs).issubset(set(mat.index)) and set(codes).issubset(set(mat.columns)):
                print(f"    两融余额矩阵命中缓存 {mat.shape}")
                return mat.loc[date_strs, codes]
        except Exception:
            pass

    bal = {}
    for d in tqdm(date_strs, desc="    读取沪市两融", leave=False):
        fp = os.path.join("margin_balance", f"{d}_margin_data.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp)
            bal[d] = dict(zip(df["标的证券代码"], df["融资余额"]))
        except Exception:
            continue
    n_sse = len(bal)
    if os.path.exists("margin_szse_cache.parquet"):
        sz = pd.read_parquet("margin_szse_cache.parquet")
        for d, g in sz.groupby("date"):
            bal.setdefault(str(d), {})
            bal[str(d)].update(dict(zip(g["code"], g["balance"])))
        print(f"    已合并深市两融 {sz['date'].nunique()} 天")

    mat = pd.DataFrame.from_dict(bal, orient="index").reindex(index=date_strs, columns=codes)
    try:
        mat.to_parquet(cache_file)
        print(f"    两融余额矩阵已缓存 {cache_file} {mat.shape}")
    except Exception:
        pass
    return mat


class DataEngine:
    """横截面数据引擎：一次加载 UNIVERSE 全部标的，对齐到共同交易日。"""

    def __init__(self):
        self.codes = []

    def load(self):
        print(f"加载 {len(UNIVERSE)} 只标的: {','.join(UNIVERSE)}")
        frames, srcs = {}, set()
        for code in tqdm(UNIVERSE, desc="    拉取行情", leave=False):
            try:
                df, s = fetch_daily(code)
                frames[code] = df
                srcs.add(s)
            except Exception as e:
                print(f"    [!] {code} 拉取失败，跳过: {e}")
        if not frames:
            raise ValueError("所有标的都拉取失败")
        self.codes = list(frames.keys())

        common = None
        for df in frames.values():
            s = set(df["date"])
            common = s if common is None else (common & s)
        common = sorted(common)
        if len(common) < 200:
            raise ValueError(f"共同交易日只有 {len(common)} 天，样本太少")
        self.dates = pd.DatetimeIndex(common)
        N, T = len(self.codes), len(common)

        open_ = np.zeros((N, T)); close_ = np.zeros((N, T)); vol_ = np.zeros((N, T))
        for i, code in enumerate(self.codes):
            df = frames[code].set_index("date").reindex(self.dates).ffill().bfill()
            open_[i] = df["open"].values
            close_[i] = df["close"].values
            vol_[i] = df["volume"].values
        self.open_np, self.close_np, self.vol_np = open_, close_, vol_
        self.raw_open = torch.from_numpy(open_.astype(np.float32)).to(DEVICE)
        self.raw_close = torch.from_numpy(close_.astype(np.float32)).to(DEVICE)
        self._src = ",".join(sorted(srcs))

        # ---- A股制度掩码 ----
        self.entry_ok = np.zeros((N, T), bool)
        self.exit_ok = np.zeros((N, T), bool)
        self.halt = np.zeros((N, T), bool)
        for i, code in enumerate(self.codes):
            self.entry_ok[i], self.exit_ok[i], self.halt[i] = build_tradability(
                open_[i], close_[i], vol_[i], code)
        print(f"    不可交易日占比: {self.halt.mean():.2%} | "
              f"开盘涨停(买不进) {self.entry_ok.mean():.2%} 可买 | "
              f"开盘跌停(卖不出) {1 - self.exit_ok.mean():.2%} 不可卖")

        # ---- 因子：全部 robust_norm ----
        full = np.zeros((N, len(_ALL_FACTORS), T), dtype=np.float32)
        for i in range(N):
            full[i] = self._features(open_[i], close_[i], vol_[i])   # 固定 6 行顺序
        mi = _ALL_FACTORS.index("MARGIN_NET")
        if "MARGIN_NET" not in FEATURES:
            print("    已跳过两融因子（FACTORS 未包含 MARGIN_NET）")
        try:
            if "MARGIN_NET" not in FEATURES:
                raise StopIteration
            # 【修复】统一用「融资余额变化率」：沪市（自带缓存）+ 深市（fetch_margin_szse.py）都能算。
            # v1 用的是「融资买入额 - 融资偿还额」，深交所明细没有「偿还额」字段，两市无法统一。
            mat = load_margin_balance_matrix(self.codes, self.dates.strftime("%Y%m%d").tolist())
            bal = mat.astype(np.float64)
            chg = bal.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            chg = chg.clip(-0.5, 0.5)                     # 去掉极端值（增发/数据修正）
            mr = chg.values.T                             # [N, T]
            for i in range(N):
                full[i, mi, :] = robust_norm(mr[i])
            valid_cols = (bal.notna().sum() > len(bal) * 0.5).sum()
            print(f"    两融余额有效标的 {valid_cols}/{N}"
                  + (f"  ← 部分标的缺两融数据（该因子对其恒为 0）" if valid_cols < N else ""))
            for i, code in enumerate(self.codes):
                if bal.iloc[:, i].notna().sum() <= len(bal) * 0.5:
                    print(f"      [缺两融] {code}")
        except StopIteration:
            pass
        except Exception as e:
            print(f"    [!] 两融矩阵失败，该因子置 0: {e}")
        sel = [_ALL_FACTORS.index(f) for f in FEATURES]
        feats = full[:, sel, :]
        self.feat_data = torch.from_numpy(feats).to(DEVICE)      # [N, F, T]
        scale = ", ".join(f"{FEATURES[k]}={feats[:, k, :].std():.2f}" for k in range(len(FEATURES)))
        print(f"    因子量级: {scale}")

        self.split_idx = int(T * 0.8)
        print(f"数据就绪 | 源={self._src} | {N} 只标的 × {T} 个交易日")
        print(f"  训练段 {self.dates[0].date()} ~ {self.dates[self.split_idx-1].date()}"
              f"（{self.split_idx} 天）")
        print(f"  样本外 {self.dates[self.split_idx].date()} ~ {self.dates[-1].date()}"
              f"（{T - self.split_idx} 天）")
        return self

    @staticmethod
    def _features(open_, close_, vol_):
        """单标的因子计算（原始 v1 的第 6 个因子漏了 robust_norm，量级 1e8，是致命 bug）。"""
        ret = np.zeros_like(close_)
        ret[1:] = (close_[1:] - close_[:-1]) / (close_[:-1] + 1e-6)
        ret5 = pd.Series(close_).pct_change(5).fillna(0).values
        vol_ma = pd.Series(vol_).rolling(20).mean().values
        vol_chg = np.zeros_like(vol_)
        m = vol_ma > 0
        vol_chg[m] = vol_[m] / vol_ma[m] - 1
        v_ret = ret * (vol_chg + 1)
        ma60 = pd.Series(close_).rolling(60).mean().values
        trend = np.zeros_like(close_)
        m = ma60 > 0
        trend[m] = close_[m] / ma60[m] - 1
        raw = [ret, ret5, vol_chg, v_ret, trend, np.zeros_like(close_)]   # 最后一个由两融矩阵填
        return np.stack([robust_norm(x) for x in raw]).astype(np.float32)

    def run_factor(self, factor, lo=0, hi=None):
        """把 [N, T] 因子跑成持仓，返回 (daily_ret [N,Tseg], flags [N,Tseg], stats)。"""
        hi = factor.shape[1] if hi is None else hi
        f = factor[:, lo:hi].detach().cpu().numpy() if torch.is_tensor(factor) \
            else np.asarray(factor)[:, lo:hi]
        sig = (np.tanh(f) > 0).astype(np.float64)
        return simulate_batch(sig, self.open_np[:, lo:hi], self.close_np[:, lo:hi],
                              self.entry_ok[:, lo:hi], self.exit_ok[:, lo:hi], HOLD_PERIOD)


# ==============================================================================
# 6. 策略搜索（REINFORCE）—— 奖励函数是原始版的核心，这里按 A股重写
# ==============================================================================
class DeepQuantMiner:
    def __init__(self, engine):
        self.engine = engine
        self.model = AlphaGPT().to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.best_score = -99.0
        self.best_formula_tokens = None

    # ---------- 语法掩码：保证生成的是合法波兰式 ----------
    def get_strict_mask(self, open_slots, step):
        B = open_slots.shape[0]
        mask = torch.full((B, VOCAB_SIZE), float("-inf"), device=DEVICE)
        remaining = MAX_SEQ_LEN - step
        done = open_slots == 0
        mask[done, 0] = 0.0
        active = ~done
        must_feat = open_slots >= remaining
        mask[active, : len(FEATURES)] = 0.0
        can_op = active & (~must_feat)
        if can_op.any():
            mask[can_op, len(FEATURES):] = 0.0
        return mask

    def solve_one(self, tokens):
        """执行波兰式公式（前缀写法，倒序遍历）→ 因子矩阵 [N 标的, T 时间]"""
        stack = []
        try:
            for t in reversed(tokens):
                if t < len(FEATURES):
                    stack.append(self.engine.feat_data[:, t, :])
                else:
                    arity = OP_ARITY_MAP[t]
                    if len(stack) < arity:
                        raise ValueError
                    args = [stack.pop() for _ in range(arity)]
                    res = OP_FUNC_MAP[t](*args)
                    if torch.isnan(res).any():
                        res = torch.nan_to_num(res)
                    stack.append(res)
            if len(stack) >= 1:
                final = stack[-1]
                if final.std() < 1e-6:
                    return None
                return final
        except Exception:
            return None
        return None

    def solve_batch(self, token_seqs):
        B = token_seqs.shape[0]
        N, T = self.engine.feat_data.shape[0], self.engine.feat_data.shape[2]
        results = torch.zeros((B, N, T), device=DEVICE)
        valid = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        for i in range(B):
            r = self.solve_one(token_seqs[i].cpu().tolist())
            if r is not None:
                results[i] = r
                valid[i] = True
        return results, valid

    # ---------- 奖励函数：Top-N 组合口径（原始版实盘 runner 的做法）----------
    def backtest(self, factors, complexity=None):
        """
        【核心】横截面 Top-N 组合打分。

        每天在标的池里挑因子值最高的 N 只等权持有（对应原始版 runner 的排序取前 N），
        持有 HOLD_PERIOD 个交易日后换股。分数 = 组合日收益的年化索提诺 − 回撤惩罚 − 复杂度惩罚。

        与旧做法的区别：旧做法是每只股票各自出二值信号，30 只里常同时买 14 只 ≈ 半个指数。
        """
        B, N, T = factors.shape
        if complexity is None:
            complexity = torch.zeros(B, device=DEVICE)
        split = self.engine.split_idx
        rewards = torch.zeros(B, device=DEVICE)

        for s in range(0, B, BATCH_CHUNK):
            chunk = factors[s:s + BATCH_CHUNK].detach().cpu().numpy()
            cb = chunk.shape[0]
            if cb == 0:
                break
            allzero = (np.abs(chunk) < 1e-12).all(axis=(1, 2))
            sub = chunk[:, :, :split]
            o = np.broadcast_to(self.engine.open_np[:, :split], (cb, N, split))
            c = np.broadcast_to(self.engine.close_np[:, :split], (cb, N, split))
            e_ok = np.broadcast_to(self.engine.entry_ok[:, :split], (cb, N, split))
            x_ok = np.broadcast_to(self.engine.exit_ok[:, :split], (cb, N, split))
            daily, st = simulate_topn_batch(sub, o, c, e_ok, x_ok,
                                            HOLD_PERIOD, TOP_N, REQUIRE_POSITIVE)

            mu = daily.mean(axis=1)
            dvol = np.sqrt((np.minimum(daily, 0.0) ** 2).mean(axis=1))
            sortino = mu / (dvol + 1e-9) * np.sqrt(252)
            eq = np.cumprod(1.0 + daily, axis=1)
            maxdd = (1.0 - eq / np.maximum.accumulate(eq, axis=1)).max(axis=1)
            n_tr = st["n_trades"]
            ex = st["exposure"]

            for b in range(cb):
                if allzero[b]:
                    rewards[s + b] = -2.0
                    continue
                # 风控门槛：交易次数下限 + 仓位暴露下限（防"几乎不持仓却比率虚高"）
                if n_tr[b] < MIN_PORTFOLIO_TRADES or ex[b] < 0.10:
                    rewards[s + b] = -3.0
                    continue
                score = float(sortino[b])
                score -= MAX_DD_PENALTY * max(0.0, float(maxdd[b]) - DD_THRESHOLD)
                score -= LEN_PENALTY * float(complexity[s + b])
                rewards[s + b] = float(np.clip(score, -5.0, 10.0))
        return rewards

    # ---------- 训练 ----------
    def train(self):
        if not FORCE_TRAIN:
            fp = self.find_best_formula_file()
            if fp and self.load_formula_from_file(fp):
                print("已加载本地公式，跳过训练（FORCE_TRAIN=False）")
                return
        print(f"Training (REINFORCE) ... MAX_LEN={MAX_SEQ_LEN} SEED={SEED}")
        pbar = tqdm(range(TRAIN_ITERATIONS))
        for _ in pbar:
            B = BATCH_SIZE
            open_slots = torch.ones(B, dtype=torch.long, device=DEVICE)
            log_probs, tokens = [], []
            curr = torch.zeros((B, 1), dtype=torch.long, device=DEVICE)
            for step in range(MAX_SEQ_LEN):
                logits, val = self.model(curr)
                mask = self.get_strict_mask(open_slots, step)
                dist = Categorical(logits=logits + mask)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                tokens.append(action)
                curr = torch.cat([curr, action.unsqueeze(1)], dim=1)
                is_op = action >= len(FEATURES)
                arity = torch.zeros(VOCAB_SIZE, dtype=torch.long, device=DEVICE)
                for k, v in OP_ARITY_MAP.items():
                    arity[k] = v
                delta = torch.full((B,), -1, device=DEVICE)
                op_delta = arity[action] - 1
                delta = torch.where(is_op, op_delta, delta)
                delta = torch.where(open_slots == 0, torch.zeros_like(delta), delta)
                open_slots = open_slots + delta
                open_slots = torch.clamp(open_slots, min=0)

            seqs = torch.stack(tokens, dim=1)
            factors, valid = self.solve_batch(seqs)
            rewards = torch.full((B,), -1.0, device=DEVICE)
            if valid.any():
                complexity = (seqs[valid] >= len(FEATURES)).sum(dim=1).float()
                scores = self.backtest(factors[valid], complexity)
                rewards[valid] = scores
                best_sub = torch.argmax(scores)
                if scores[best_sub].item() > self.best_score:
                    self.best_score = scores[best_sub].item()
                    self.best_formula_tokens = seqs[valid][best_sub].cpu().tolist()
            # 优势归一化（原始 model_core 的做法：减均值再除标准差）
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            loss = 0
            for lp in log_probs:
                loss = loss + (-lp * adv)
            loss = loss.mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            pbar.set_postfix({"Valid": f"{valid.float().mean().item():.1%}",
                              "Best": f"{self.best_score:.3f}"})
        if self.best_formula_tokens is None:
            raise RuntimeError("训练未找到任何有效公式，请放宽 MIN_TRADES 或检查数据")
        self.save_formula()

    # ---------- 公式存/取 ----------
    def find_best_formula_file(self):
        fs = glob.glob(f"{INDEX_CODE}_best_formula_v2_*.txt")
        return max(fs, key=os.path.getctime) if fs else None

    def load_formula_from_file(self, path):
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().strip().split("\n")
            self.best_score = float(lines[0].split(":", 1)[1].strip())
            self.best_formula_tokens = [int(x) for x in
                                        lines[1].split(":", 1)[1].strip().strip("[]").split(",")]
            print(f"加载公式: {path} | Score={self.best_score:.4f}")
            return True
        except Exception as e:
            print(f"[!] 加载公式失败: {e}")
            return False

    def save_formula(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{INDEX_CODE}_best_formula_v2_{ts}.txt"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"BestScore: {self.best_score:.4f}\n")
            fh.write(f"Tokens: {self.best_formula_tokens}\n")
            fh.write(f"Formula: {self.decode()}\n")
        print(f"公式已保存: {path}")

    def decode(self, tokens=None):
        tokens = self.best_formula_tokens if tokens is None else tokens
        if not tokens:
            return "N/A"
        stream = list(tokens)

        def _p():
            if not stream:
                return ""
            t = stream.pop(0)
            if t < len(FEATURES):
                return FEATURES[t]
            return f"{VOCAB[t]}({','.join(_p() for _ in range(OP_ARITY_MAP[t]))})"

        try:
            return _p()
        except Exception:
            return "Invalid"

    def encode(self, formula_str):
        """把可读公式（如 SUB(NEG(TREND),RET)）反解成 token 序列。"""
        import re
        raw = re.findall(r"[A-Z0-9_]+|\(|\)|,", formula_str.upper())
        vmap = {n: i for i, n in enumerate(VOCAB)}
        pos = 0

        def parse():
            nonlocal pos
            if pos >= len(raw):
                raise ValueError("公式不完整")
            name = raw[pos]
            pos += 1
            idx = vmap.get(name)
            if idx is None:
                raise ValueError(f"未知 token: {name}")
            if idx < len(FEATURES):
                return [idx]
            if pos >= len(raw) or raw[pos] != "(":
                raise ValueError(f"算子 {name} 缺少参数")
            pos += 1
            arity = OP_ARITY_MAP[idx]
            kids = []
            for a in range(arity):
                kids.extend(parse())
                if a < arity - 1:
                    if pos >= len(raw) or raw[pos] != ",":
                        raise ValueError(f"{name} 参数分隔符错误")
                    pos += 1
            if pos >= len(raw) or raw[pos] != ")":
                raise ValueError(f"{name} 缺少右括号")
            pos += 1
            return [idx] + kids

        tokens = parse()
        if pos != len(raw):
            raise ValueError(f"公式尾部有无法解析的内容: {raw[pos:]}")
        return tokens


# ==============================================================================
# 7. 样本外检验（Top-N 组合口径）
# ==============================================================================
def _name(code):
    return f"{code} {STOCK_NAMES[code]}" if code in STOCK_NAMES else code


def _oos_portfolio(engine, factor):
    """在样本外段跑 Top-N 组合，返回 (daily_ret, stats, dates)"""
    split = engine.split_idx
    N = len(engine.codes)
    sub = factor[:, split:].detach().cpu().numpy()[None, :, :]
    T = sub.shape[2]
    o = np.broadcast_to(engine.open_np[:, split:], (1, N, T))
    c = np.broadcast_to(engine.close_np[:, split:], (1, N, T))
    e_ok = np.broadcast_to(engine.entry_ok[:, split:], (1, N, T))
    x_ok = np.broadcast_to(engine.exit_ok[:, split:], (1, N, T))
    daily, st = simulate_topn_batch(sub, o, c, e_ok, x_ok,
                                    HOLD_PERIOD, TOP_N, REQUIRE_POSITIVE)
    sq = {k: (v[0] if hasattr(v, "shape") and v.ndim >= 1 else v) for k, v in st.items()}
    return daily[0], sq, engine.dates[split:]


def final_reality_check(engine, miner):
    print("\n" + "=" * 74)
    print("样本外检验（Out-of-Sample，Top-N 组合口径，训练段完全未参与搜索）")
    print("=" * 74)
    print(f"策略公式: {miner.decode()}")
    print(f"标的池  : {len(engine.codes)} 只　持有 {HOLD_PERIOD} 日　每次持仓 {TOP_N} 只（等权）")
    factor = miner.solve_one(miner.best_formula_tokens)
    if factor is None:
        print("公式无法执行")
        return None

    port, st, dates = _oos_portfolio(engine, factor)
    pst = perf_stats(port)

    closes = engine.close_np[:, engine.split_idx:]
    bh = np.zeros(len(port))
    bh[1:] = (closes[:, 1:] / closes[:, :-1] - 1.0).mean(axis=0)
    bh_st = perf_stats(bh)

    print("-" * 74)
    print(f"样本外区间 : {dates[0].date()} ~ {dates[-1].date()}（{len(dates)} 个交易日）")
    print(f"{'指标':<16}{'Top-N 组合':>16}{'等权买入持有':>18}")
    for label, key, fmt in [
        ("总收益", "total", "{:.2%}"), ("年化收益", "ann", "{:.2%}"),
        ("年化波动", "vol", "{:.2%}"), ("夏普", "sharpe", "{:.2f}"),
        ("索提诺", "sortino", "{:.2f}"), ("最大回撤", "max_dd", "{:.2%}"),
        ("卡玛", "calmar", "{:.2f}"),
    ]:
        print(f"{label:<16}{fmt.format(pst[key]):>16}{fmt.format(bh_st[key]):>18}")
    n_tr = int(st["n_trades"]); n_win = int(st["wins"])
    print("-" * 74)
    if n_tr:
        print(f"交易 {n_tr} 笔　胜率 {n_win / n_tr:.1%}　"
              f"平均单笔净收益 {float(st['avg_net']):.2%}　仓位暴露 {float(st['exposure']):.0%}　"
              f"因跌停/停牌顺延 {int(st['forced'])} 次")
    else:
        print(f"样本外无交易（合格标的长期少于 MIN_PICKS={MIN_PICKS}）　"
              f"仓位暴露 {float(st['exposure']):.0%}")
    picks = st["pick_count"]
    order = np.argsort(-picks)[:12]
    print("被选中次数最多的标的: " +
          "、".join(f"{_name(engine.codes[i])}×{int(picks[i])}" for i in order if picks[i] > 0))
    print("=" * 74)
    print("说明：Top-N 等权组合、日线级别；不含仓位动态调整与行业中性；历史回测不代表未来。")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                                 gridspec_kw={"width_ratios": [1.7, 1]})
        axes[0].plot(dates, pst["equity"], label=f"Top-{TOP_N} Strategy")
        axes[0].plot(dates, bh_st["equity"], label="Equal-weight Buy & Hold", alpha=0.7)
        axes[0].set_title(f"OOS: Ann {pst['ann']:.1%} | Sortino {pst['sortino']:.2f} | "
                          f"MaxDD {pst['max_dd']:.1%}")
        axes[0].grid(alpha=0.3); axes[0].legend()
        dd = 1.0 - pst["equity"] / np.maximum.accumulate(pst["equity"])
        axes[1].fill_between(dates, -dd, 0, alpha=0.5, color="crimson")
        axes[1].set_title("Drawdown")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("strategy_performance_v2.png", dpi=110)
        print("净值图已保存: strategy_performance_v2.png")
    except Exception as e:
        print(f"[!] 绘图失败: {e}")
    return pst, st


# ==============================================================================
# 8. 每日信号推送
# ==============================================================================
def next_trading_day(after):
    """返回 after 之后的第一个交易日（含节假日）。失败则回退为下一个工作日。"""
    try:
        bs = _bs_login_once()
        start = (after + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (after + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        while rs.error_code == "0" and rs.next():
            d, flag = rs.get_row_data()
            if flag == "1":
                return pd.Timestamp(d)
    except Exception:
        pass
    d = after + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def report_latest(engine, miner, n_days=10):
    """每日推送：Top-N 选股名单 + 组合样本外表现"""
    factor = miner.solve_one(miner.best_formula_tokens)
    port, st, dates = _oos_portfolio(engine, factor)
    pst = perf_stats(port)
    f_np = factor.detach().cpu().numpy()
    N = len(engine.codes)

    last_date = pd.Timestamp(dates[-1])
    exec_date = next_trading_day(last_date)
    today = pd.Timestamp(datetime.today().date())
    if last_date < today - pd.Timedelta(days=1) and exec_date <= today:
        print(f"\n[!] 数据日期 {last_date.date()}，执行日 {exec_date.date()} 可能已过，请检查数据源")

    # ---- 下一批选股（复现模拟器的决策逻辑）----
    sig_last = np.tanh(f_np[:, -1])
    elig = engine.entry_ok[:, -1].copy()
    if REQUIRE_POSITIVE:
        elig = elig & (sig_last > 0)
    score = np.where(elig, sig_last, -np.inf)
    order = np.argsort(-score)[:TOP_N]
    picks = [int(i) for i in order if elig[i]]

    print(f"\n{'='*74}\n信号：{last_date.date()} 收盘　→　执行日 {exec_date.date()} 开盘"
          f"（Top-{TOP_N} 等权）\n{'='*74}")
    print(f"{'排名':<6}{'标的':<18}{'因子值':>10}{'当日可买':>10}")
    for rank, i in enumerate(picks, 1):
        print(f"{rank:<6}{_name(engine.codes[i]):<18}{f_np[i, -1]:>10.3f}"
              f"{'是' if engine.entry_ok[i, -1] else '否':>10}")
    if not picks:
        print("（无满足条件的标的 → 空仓）")
    print("-" * 74)
    n_tr = int(st["n_trades"]); n_win = int(st["wins"])
    print(f"样本外组合：累计 {pst['total']:.2%} | 年化 {pst['ann']:.2%} | "
          f"索提诺 {pst['sortino']:.2f} | 最大回撤 {pst['max_dd']:.2%}")
    print(f"　　　　　　　交易 {n_tr} 笔 | 胜率 {n_win / n_tr:.1%} | 仓位暴露 {st['exposure']:.0%}"
          if n_tr else "　　　　　　　样本外无交易")

    # ---- 钉钉 ----
    lines = [f"## 📊 AlphaGPT Top-{TOP_N} [{len(engine.codes)}只池]", ""]
    lines.append(f"**信号：{last_date.date()} 收盘**  →  **执行：{exec_date.date()} 开盘**")
    if picks:
        lines.append("")
        lines.append(f"### 买入名单（等权 {100.0 / len(picks):.0f}% / 只）")
        for rank, i in enumerate(picks, 1):
            lines.append(f"{rank}. **{_name(engine.codes[i])}**　因子值 {f_np[i, -1]:.3f}")
        lines.append("")
        lines.append(f"共 {len(picks)}/{len(engine.codes)} 只入选")
    else:
        lines.append("")
        lines.append("⬜ **无入选标的 → 空仓**")
    lines.append("")
    lines.append(f"**公式**：`{miner.decode()}`　**持有**：{HOLD_PERIOD} 个交易日")
    lines.append("")
    lines.append("### 📈 样本外组合")
    lines.append(f"- 累计收益 **{pst['total']:.2%}**　年化 {pst['ann']:.2%}")
    lines.append(f"- 索提诺 {pst['sortino']:.2f}　夏普 {pst['sharpe']:.2f}　"
                 f"最大回撤 {pst['max_dd']:.2%}")
    if n_tr:
        lines.append(f"- 交易 {n_tr} 笔　胜率 {n_win / n_tr:.1%}　仓位暴露 {st['exposure']:.0%}")
    lines.append("")
    lines.append("> 已含佣金/过户费/印花税/滑点；开盘涨停不买入、跌停顺延、停牌跳过。")
    if pst["total"] < 0:
        lines.append(">")
        lines.append("> ⚠️ **研究性质，非投资建议**：该公式在样本外为负收益、跑输等权买入持有，"
                     "尚未验证出稳定超额收益。")
    send_dingtalk_msg("\n".join(lines))
    return pst, st


# ==============================================================================
# 9. 两融数据（沿用 v1：按日缓存 + 增量抓取）
# ==============================================================================
def get_margin_balance(stock_code, date_list):
    cache_dir = "margin_balance"
    os.makedirs(cache_dir, exist_ok=True)
    margin_data, missing = {}, []
    for date in date_list:
        fp = os.path.join(cache_dir, f"{date}_margin_data.parquet")
        if os.path.exists(fp):
            try:
                df = pd.read_parquet(fp)
                rows = df[df["标的证券代码"] == stock_code]
                if not rows.empty:
                    margin_data[date] = rows.iloc[0].to_dict()
                    continue
            except Exception:
                pass
        missing.append(date)
    today = datetime.today().strftime("%Y%m%d")
    missing = [d for d in missing if d != today]
    if missing:
        print(f"    两融缺失 {len(missing)} 天，开始抓取...")
        margin_data.update(_fetch_margin_data(INDEX_CODE, missing, cache_dir))
    else:
        print(f"    两融数据全部命中缓存（{len(margin_data)} 天）")

    out = {k: [] for k in ("bal", "buy", "repay", "short")}
    for date in date_list:
        row = margin_data.get(date, {})
        out["bal"].append(float(row.get("融资余额", 0)))
        out["buy"].append(float(row.get("融资买入额", 0)))
        out["repay"].append(float(row.get("融资偿还额", 0)))
        out["short"].append(float(row.get("融券余量", 0)))
    return tuple(torch.tensor(out[k], dtype=torch.float32, device=DEVICE)
                 for k in ("bal", "buy", "repay", "short"))


def _fetch_margin_data(stock_code, date_list, cache_dir):
    """注意：仓库缓存是「沪市」两融（stock_margin_detail_sse），深市标的会取不到。"""
    result = {}
    failed = 0
    for date in tqdm(date_list, desc="Fetching margin data", leave=False):
        try:
            import akshare as ak
            df = ak.stock_margin_detail_sse(date=date)
            if df is None or df.empty:
                failed += 1
                continue
            df.to_parquet(os.path.join(cache_dir, f"{date}_margin_data.parquet"))
            rows = df[df["标的证券代码"] == stock_code]
            if not rows.empty:
                r = rows.iloc[0]
                result[date] = {
                    "融资余额": float(r.get("融资余额", 0)),
                    "融资买入额": float(r.get("融资买入额", 0)),
                    "融资偿还额": float(r.get("融资偿还额", 0)),
                    "融券余量": float(r.get("融券余量", 0)),
                }
        except Exception:
            failed += 1
    if failed:
        print(f"    {failed} 天无两融数据（非两融标的/深市/非交易日）")
    return result


# ==============================================================================
# 10. 主流程
# ==============================================================================
def main(realitytest=True):
    print("=" * 72)
    print(f"AlphaGPT-Routine v2 | 标的池 {len(UNIVERSE)} 只 | 持有 {HOLD_PERIOD} 交易日 | "
          f"买/卖成本 {BUY_COST:.4%}/{SELL_COST:.4%}")
    print("=" * 72)
    eng = DataEngine().load()
    miner = DeepQuantMiner(eng)
    if BEST_FORMULA:
        print(f"使用指定公式（跳过训练）: {BEST_FORMULA}")
        miner.best_formula_tokens = miner.encode(BEST_FORMULA)
    else:
        miner.train()
    if realitytest:
        final_reality_check(eng, miner)
    report_latest(eng, miner, n_days=LAST_NDAYS)
    return eng, miner


if __name__ == "__main__":
    CODE_FORMULA = _get_env("CODE_FORMULA", "")
    if not CODE_FORMULA:
        main(realitytest=True)
    else:
        # 兼容旧用法：CODE_FORMULA="600519:ADD(RET,TREND)"，每行一组成对运行（单标的）
        for cf in CODE_FORMULA.split("\n"):
            if ":" not in cf:
                continue
            code, formula = cf.split(":", 1)
            UNIVERSE = [c.strip() for c in code.split(",") if c.strip()]
            INDEX_CODE = UNIVERSE[0]
            BEST_FORMULA = formula.strip()
            print(f"code={UNIVERSE} formula={BEST_FORMULA}")
            main(realitytest=True)
