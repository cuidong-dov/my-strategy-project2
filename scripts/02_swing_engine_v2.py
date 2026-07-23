#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股板块评估系统 · 小波段策略引擎 v2
新增指标：量价分析、回调到位、MACD背离、量价背离
- 默认模式：用已有最优参数快速生成信号（日常使用）
- 优化模式：两轮网格搜索寻找最优参数（偶尔运行）
"""
import pandas as pd, numpy as np, json, os, itertools, statistics as st, time, sys, argparse

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

SECTORS = ["半导体", "消费电子", "通信", "电池", "电力", "煤炭"]
MARKET_FILE = "index_沪深300.csv"

# ============ 指标计算 ============
def pct_rank(s, w=250):
    return s.rolling(w, min_periods=max(20, w//4)).apply(
        lambda x: (x[-1] <= x).mean() * 100.0, raw=True)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def add_indicators(df):
    """v2版本：在原v1基础上新增量价分析、回调到位、MACD背离"""
    df = df.copy().sort_values("date").reset_index(drop=True)
    c = df["close"]; v = df["volume"]; h = df["high"]; l = df["low"]; o = df["open"]

    # ---- v1 基础指标 ----
    df["ret"] = c.pct_change()
    for n in (5, 10, 20, 60, 120, 250):
        df[f"ma{n}"] = c.rolling(n, min_periods=max(3, n//2)).mean()
    df["disp_ma5"]   = c / df["ma5"]   - 1
    df["disp_ma20"]  = c / df["ma20"]  - 1
    df["disp_ma60"]  = c / df["ma60"]  - 1
    df["disp_ma250"] = c / df["ma250"] - 1
    df["dd_5"]  = c / c.rolling(5).max()  - 1
    df["dd_20"] = c / c.rolling(20).max() - 1
    df["vol_ma20"] = v.rolling(20, min_periods=10).mean()
    df["vol_ratio"] = v / df["vol_ma20"]
    vol20 = df["ret"].rolling(20, min_periods=10).std()
    vm = vol20.rolling(250, min_periods=60).mean()
    vs = vol20.rolling(250, min_periods=60).std()
    df["vol_z"] = (vol20 - vm) / vs

    # 五大字段（全局250日分位）
    df["板块风险"] = pct_rank(0.6 * df["disp_ma250"] + 0.4 * df["vol_z"], 250).round(1)
    weak = (df["disp_ma20"].clip(upper=0) / -0.15).clip(0, 1)
    df["抄底狂热"] = pct_rank(df["vol_ratio"] * (0.3 + 0.7 * weak), 250).round(1)
    strong = (df["disp_ma20"].clip(lower=0) / 0.15).clip(0, 1)
    df["追涨狂热"] = pct_rank(df["vol_ratio"] * (0.3 + 0.7 * strong), 250).round(1)
    panic_raw = ((df["dd_20"].clip(upper=0) / -0.20).clip(0, 1) * 0.6
               + (df["vol_ratio"].clip(upper=3) - 1).clip(0, 2) / 2 * 0.2
               + (df["ret"].clip(upper=0) / -0.04).clip(0, 1) * 0.2)
    df["恐慌值"] = pct_rank(panic_raw, 250).round(1)
    df["恐慌值_局部"]  = pct_rank(panic_raw, 20).round(1)
    df["恐慌值_60日"]  = pct_rank(panic_raw, 60).round(1)  # 中期恐慌（评分用）
    df["板块风险_局部"] = pct_rank(0.6 * df["disp_ma250"] + 0.4 * df["vol_z"], 20).round(1)
    df["close_prev"]  = c.shift(1)
    df["price_up"]    = (c > df["close_prev"]).astype(int)
    df["vol_shrink"]  = (df["vol_ratio"] < 1.0).astype(int)
    df["settle"]      = ((df["price_up"] == 1) & (df["vol_shrink"] == 1)).astype(int)
    df["oversold_local"] = ((-df["dd_5"]).clip(0, 0.10) / 0.10).clip(0, 1)

    # ============ v2 新增指标 ============

    # ---- 1. 量价分析 ----
    # 1a. 下跌日的量能特征：放量跌=恐慌出逃(差)，缩量跌=抛压枯竭(好)
    df["down_day"] = (df["ret"] < 0).astype(int)
    df["vol_heavy_down"] = ((df["down_day"] == 1) & (df["vol_ratio"] > 1.3)).astype(int)  # 放量下跌
    df["vol_light_down"] = ((df["down_day"] == 1) & (df["vol_ratio"] < 0.8)).astype(int)  # 缩量下跌
    # 近5日放量下跌天数占比
    df["heavy_down_5d"] = df["vol_heavy_down"].rolling(5).sum() / 5

    # 1b. 量价配合度：涨放量+跌缩量 = 健康
    up_vol = (df["ret"] > 0).astype(int) * df["vol_ratio"]
    dn_vol = (df["ret"] < 0).astype(int) * df["vol_ratio"]
    df["vol_quality"] = (up_vol.rolling(10).mean() - dn_vol.rolling(10).mean()) / df["vol_ratio"].rolling(10).mean()
    df["vol_quality"] = df["vol_quality"].clip(-2, 2)

    # 1c. 缩量程度（连续缩量天数）
    df["vol_shrink_streak"] = 0
    streak = 0
    for i in range(len(df)):
        if df["vol_shrink"].iloc[i] == 1:
            streak += 1
        else:
            streak = 0
        df.iloc[i, df.columns.get_loc("vol_shrink_streak")] = streak

    # ---- 2. 回调到位 ----
    # 2a. 从最近20日高点的回调幅度（斐波那契参考）
    roll20_high = c.rolling(20).max()
    df["pullback_from_high"] = c / roll20_high - 1  # 负值=回调中

    # 2b. 最近一波上涨的涨幅和回调比例
    # 用简化方法：找最近20日内的最低点和之后的最高点
    roll20_low = l.rolling(20).min()
    roll20_high_idx = h.rolling(20).apply(lambda x: np.argmax(x), raw=True)
    # 回调到斐波那契0.618位的距离
    df["fib_618"] = roll20_high - (roll20_high - roll20_low) * 0.618
    df["near_fib_618"] = (c < df["fib_618"] * 1.02).astype(int)  # 接近0.618支撑
    df["fib_500"] = roll20_high - (roll20_high - roll20_low) * 0.500
    df["near_fib_500"] = (c < df["fib_500"] * 1.02).astype(int)  # 接近0.5支撑
    df["fib_382"] = roll20_high - (roll20_high - roll20_low) * 0.382
    df["near_fib_382"] = (c < df["fib_382"] * 1.02).astype(int)  # 接近0.382支撑

    # 2c. 是否接近前低（二次探底）
    df["prev_low_20"] = l.rolling(20).min().shift(1)
    df["near_prev_low"] = ((c - df["prev_low_20"]) / df["prev_low_20"] < 0.03).astype(int)

    # 2d. 回调是否充分：回调幅度 > 近期波动率的1.5倍
    atr14 = (h - l).rolling(14).mean()
    df["pullback_deep"] = ((-df["pullback_from_high"]) > (atr14 * 1.5 / c)).astype(int)

    # ---- 3. MACD 背离 ----
    # 3a. MACD 标准计算
    ema12 = ema(c, 12); ema26 = ema(c, 26)
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = ema(df["macd_dif"], 9)
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # 3b. MACD 底背离：价格创20日新低，但MACD DIF没有创20日新低
    price_20low = c == c.rolling(20).min()
    dif_20low = df["macd_dif"] == df["macd_dif"].rolling(20).min()
    # 价格新低但DIF没新低 = 底背离
    df["macd_bull_div"] = 0
    for i in range(40, len(df)):
        if price_20low.iloc[i]:
            # 找上一个价格低点（20日内）
            prev_window = df.iloc[i-20:i]
            prev_lows = prev_window[prev_window["close"] == prev_window["close"].rolling(5).min()]
            if len(prev_lows) > 0:
                prev_low_dif = prev_lows["macd_dif"].iloc[-1]
                if df["macd_dif"].iloc[i] > prev_low_dif:
                    df.iloc[i, df.columns.get_loc("macd_bull_div")] = 1

    # 3c. MACD 顶背离：价格创20日新高，但MACD DIF没有创20日新高
    price_20high = c == c.rolling(20).max()
    df["macd_bear_div"] = 0
    for i in range(40, len(df)):
        if price_20high.iloc[i]:
            prev_window = df.iloc[i-20:i]
            prev_highs = prev_window[prev_window["close"] == prev_window["close"].rolling(5).max()]
            if len(prev_highs) > 0:
                prev_high_dif = prev_highs["macd_dif"].iloc[-1]
                if df["macd_dif"].iloc[i] < prev_high_dif:
                    df.iloc[i, df.columns.get_loc("macd_bear_div")] = 1

    # 3d. 量价底背离：价格创20日新低，但成交量比上一波低点更小（抛压枯竭）
    df["vol_bull_div"] = 0
    for i in range(40, len(df)):
        if price_20low.iloc[i]:
            prev_window = df.iloc[i-20:i]
            prev_lows = prev_window[prev_window["close"] == prev_window["close"].rolling(5).min()]
            if len(prev_lows) > 0:
                prev_low_vol = prev_lows["volume"].iloc[-1]
                if df["volume"].iloc[i] < prev_low_vol * 0.85:  # 量缩15%以上
                    df.iloc[i, df.columns.get_loc("vol_bull_div")] = 1

    # 3e. 量价顶背离：价格创20日新高，但成交量比上一波高点更小（追涨无力）
    df["vol_bear_div"] = 0
    for i in range(40, len(df)):
        if price_20high.iloc[i]:
            prev_window = df.iloc[i-20:i]
            prev_highs = prev_window[prev_window["close"] == prev_window["close"].rolling(5).max()]
            if len(prev_highs) > 0:
                prev_high_vol = prev_highs["volume"].iloc[-1]
                if df["volume"].iloc[i] < prev_high_vol * 0.85:
                    df.iloc[i, df.columns.get_loc("vol_bear_div")] = 1

    # 3f. MACD金叉/死叉
    df["macd_golden"] = ((df["macd_dif"] > df["macd_dea"]) & (df["macd_dif"].shift(1) <= df["macd_dea"].shift(1))).astype(int)
    df["macd_dead"]   = ((df["macd_dif"] < df["macd_dea"]) & (df["macd_dif"].shift(1) >= df["macd_dea"].shift(1))).astype(int)

    # 3g. MACD 趋势强度
    df["macd_trend"] = df["macd_dif"] - df["macd_dif"].shift(5)  # 5日DIF变化方向

    # ---- 4. 趋势健康度（方案A） ----
    # 4a. 均线斜率
    df["ma5_slope"]  = df["ma5"]  / df["ma5"].shift(5)  - 1   # MA5的5日斜率
    df["ma20_slope"] = df["ma20"] / df["ma20"].shift(10) - 1   # MA20的10日斜率
    # 4b. 均线排列：多头=1，空头=-1，交叉/走平=0
    df["ma_align"] = 0
    df.loc[(df["ma5"] > df["ma20"]) & (df["ma20"] > df["ma60"]), "ma_align"] = 1
    df.loc[(df["ma5"] < df["ma20"]) & (df["ma20"] < df["ma60"]), "ma_align"] = -1
    # 4c. 连续下跌天数
    df["down_streak"] = 0
    streak = 0
    for i in range(len(df)):
        if df["ret"].iloc[i] < 0:
            streak += 1
        else:
            streak = 0
        df.iloc[i, df.columns.get_loc("down_streak")] = streak
    # 4d. 价格在均线系统中的位置（-1=所有均线下方，1=所有均线上方）
    above_ma5 = (c > df["ma5"]).astype(int)
    above_ma20 = (c > df["ma20"]).astype(int)
    above_ma60 = (c > df["ma60"]).astype(int)
    df["price_vs_ma"] = (above_ma5 + above_ma20 + above_ma60 - 1.5) / 1.5  # 归一化到-1~1

    # ---- 5. 支撑位距离（方案C） ----
    # 5a. 距60日均线距离（百分比）
    df["dist_to_ma60"] = c / df["ma60"] - 1
    # 5b. 距前低的距离
    df["dist_to_prev_low"] = (c - df["prev_low_20"]) / df["prev_low_20"]
    # 5c. 综合支撑强度：同时靠近60日线和前低=强支撑
    near_ma60 = (abs(df["dist_to_ma60"]) < 0.05).astype(int)
    near_low  = (df["dist_to_prev_low"] < 0.03).astype(int)
    df["support_score"] = near_ma60 + near_low  # 0=无支撑, 1=单一支撑, 2=双重支撑

    return df

def add_market_risk(df, mkt):
    m = add_indicators(mkt)
    mr = m[["date", "板块风险"]].rename(columns={"板块风险": "大盘风险"})
    df = df.merge(mr, on="date", how="left")
    df["大盘风险"] = df["大盘风险"].ffill().bfill()
    return df

# ============ 买入成功率（v2 八因子版） ============
def add_buy_success_prob_5d(df):
    """
    v2版本：8因子分箱法
    新增3个因子：vol_quality(量价质量)、pullback_from_high(回调深度)、macd_trend(MACD趋势)
    """
    n = len(df)
    prob = np.full(n, 50.0)
    close = df["close"].values

    feats = np.column_stack([
        df["恐慌值_局部"].values,
        df["板块风险_局部"].values,
        df["抄底狂热"].values,
        df["vol_ratio"].values,
        (df["disp_ma5"] * 100).values,
        df["vol_quality"].fillna(0).values,        # v2新增
        (df["pullback_from_high"] * 100).values,    # v2新增
        df["macd_trend"].fillna(0).values,          # v2新增
    ])
    n_feat = feats.shape[1]
    bins = 5
    win = 120

    fwd_up = np.zeros(n, dtype=int)
    for i in range(n - 5):
        fwd_up[i] = 1 if close[i + 5] > close[i] else 0

    for i in range(win, n):
        w = feats[i - win:i]
        w_fwd = fwd_up[i - win:i - 5]
        w = w[:len(w_fwd)]
        if len(w) < 30:
            continue

        bin_ids = np.zeros((len(w), n_feat), dtype=int)
        cur_bin = np.zeros(n_feat, dtype=int)
        for k in range(n_feat):
            col = w[:, k]
            edges = np.percentile(col, np.linspace(0, 100, bins + 1))
            edges = np.unique(np.round(edges, 1))
            if len(edges) < 3:
                continue
            ids = np.digitize(col, edges[1:-1])
            bin_ids[:, k] = np.clip(ids, 0, bins - 1)
            cid = np.digitize([feats[i, k]], edges[1:-1])[0]
            cur_bin[k] = np.clip(cid, 0, bins - 1)

        match = np.ones(len(w), dtype=bool)
        for k in range(n_feat):
            match = match & (bin_ids[:, k] == cur_bin[k])

        if match.sum() >= 3:
            prob[i] = round(w_fwd[match].mean() * 100, 1)

    df["买入预估成功率_5d"] = prob
    df["买入预估成功率_5d"] = df["买入预估成功率_5d"].ffill().bfill().clip(0, 100).round(1)
    return df

# ============ 买点评分系统 ============
def score_buy_point(row, return_detail=False, trend_mode=False):
    """
    买点1-10分综合评分。
    return_detail=True 时返回 (总分, 各维度原始分dict)
    trend_mode=True 时使用趋势买点权重（回调降权，趋势升权）
    """
    dims = {}  # 各维度原始分（1-10）

    # 维度1：回调深度
    pullback = -row.get("pullback_from_high", 0)
    if pullback > 0.35:       dims["回调深度"] = 10.0
    elif pullback > 0.30:     dims["回调深度"] = 9.0
    elif pullback > 0.25:     dims["回调深度"] = 8.0
    elif pullback > 0.20:     dims["回调深度"] = 7.0
    elif pullback > 0.15:     dims["回调深度"] = 5.5
    elif pullback > 0.12:     dims["回调深度"] = 4.5
    elif pullback > 0.08:     dims["回调深度"] = 3.0
    elif pullback > 0.05:     dims["回调深度"] = 2.0
    elif pullback > 0.03:     dims["回调深度"] = 1.5
    else:                     dims["回调深度"] = 1.0

    # 维度2：背离确认
    macd_div = row.get("macd_bull_div", 0)
    vol_div  = row.get("vol_bull_div", 0)
    golden   = row.get("macd_golden", 0)
    dif_val  = row.get("macd_dif", 0)
    if macd_div and vol_div:   dims["背离确认"] = 10.0
    elif macd_div:             dims["背离确认"] = 8.0
    elif vol_div:              dims["背离确认"] = 6.5
    elif golden == 1:          dims["背离确认"] = 4.5
    elif dif_val > 0:          dims["背离确认"] = 3.0
    else:                      dims["背离确认"] = 1.5

    # 维度3：量价健康度
    vol_q = row.get("vol_quality", 0)
    vol_r = row.get("vol_ratio", 1.0)
    settle = row.get("settle", 0)
    heavy  = row.get("heavy_down_5d", 0)
    vs = 5.0
    if vol_q > 0.8:   vs += 3.0
    elif vol_q > 0.3: vs += 1.5
    elif vol_q < -0.5: vs -= 2.0
    if settle == 1:   vs += 2.0
    if vol_r < 0.6:   vs += 1.5
    elif vol_r < 0.8: vs += 0.5
    if heavy > 0.6:   vs -= 3.0
    elif heavy > 0.4: vs -= 1.5
    dims["量价健康度"] = max(1.0, min(10.0, vs))

    # 维度4：趋势健康度
    ma_align = row.get("ma_align", 0)
    ma5_s = row.get("ma5_slope", 0)
    down_s = row.get("down_streak", 0)
    price_vs = row.get("price_vs_ma", 0)
    trend = 5.0
    if ma_align == 1:   trend += 2.0
    elif ma_align == -1: trend -= 2.0
    if ma5_s > 0.005:   trend += 1.5
    elif ma5_s < -0.005: trend -= 1.5
    if price_vs > 0.3:  trend += 1.0
    elif price_vs < -0.3: trend -= 1.0
    if down_s >= 5:     trend -= 2.0
    elif down_s >= 3:   trend -= 1.0
    dims["趋势健康度"] = max(1.0, min(10.0, trend))

    # 维度5：支撑位强度
    support = row.get("support_score", 0)
    dist_ma60 = abs(row.get("dist_to_ma60", 0.2))
    if support == 2:        dims["支撑位强度"] = 10.0
    elif support == 1:      dims["支撑位强度"] = 7.5
    elif support == 0:
        if dist_ma60 < 0.05:   dims["支撑位强度"] = 5.0
        else:                  dims["支撑位强度"] = 3.5
    elif support == -1:     dims["支撑位强度"] = 2.0
    else:                   dims["支撑位强度"] = 1.0

    # 维度6：恐慌极端度
    panic_l = row.get("恐慌值_局部", 50)
    panic_60 = row.get("恐慌值_60日", 50)
    panic = panic_l * 0.3 + panic_60 * 0.7
    if panic > 85:     dims["恐慌极端度"] = 10.0
    elif panic > 70:   dims["恐慌极端度"] = 8.0
    elif panic > 55:   dims["恐慌极端度"] = 6.0
    elif panic > 40:   dims["恐慌极端度"] = 4.0
    elif panic > 25:   dims["恐慌极端度"] = 2.5
    else:              dims["恐慌极端度"] = 1.5

    # 维度7：企稳确认度
    shrink_streak = row.get("vol_shrink_streak", 0)
    if row.get("settle", 0) == 1:
        if shrink_streak >= 4:   dims["企稳确认度"] = 10.0
        elif shrink_streak >= 3: dims["企稳确认度"] = 8.0
        elif shrink_streak >= 2: dims["企稳确认度"] = 6.0
        else:                    dims["企稳确认度"] = 4.0
    else:
        dims["企稳确认度"] = 2.0

    # 加权求和（趋势模���权重不同）
    if trend_mode:
        weights = {"回调深度":0.15, "背离确认":0.15, "量价健康度":0.20,
                   "趋势健康度":0.25, "支撑位强度":0.10, "恐慌极端度":0.05, "企稳确认度":0.10}
    else:
        weights = {"回调深度":0.30, "背离确认":0.20, "量价健康度":0.15,
                   "趋势健康度":0.15, "支撑位强度":0.10, "恐慌极端度":0.05, "企稳确认度":0.05}
    total = round(sum(dims[k] * weights[k] for k in dims), 1)

    if return_detail:
        return total, dims
    return total

def hard_filter(row):
    """
    硬过滤：满足任一条件直接拒绝买入信号。
    返回 True = 通过过滤（允许买入），False = 拒绝。
    """
    # 条件1：连续放量下跌 ≥ 4天
    if row.get("heavy_down_5d", 0) >= 0.8 and row.get("down_streak", 0) >= 4:
        return False
    # 条件2：MA全空头 + 价格在MA20下方 > 5% + 加速下跌
    if (row.get("ma_align", 0) == -1 and
        row.get("disp_ma20", 0) < -0.05 and
        row.get("ma5_slope", 0) < -0.01):
        return False
    # 条件3：当日暴跌 > 7%
    if row.get("ret", 0) < -0.07:
        return False
    return True

# ============ 信号与回测 ============
def gen_signals(df, p):
    df = df.copy()
    risk = df["板块风险"]; mkt = df["大盘风险"]
    risk_l = df["板块风险_局部"]; panic_l = df["恐慌值_局部"]
    panic_60 = df.get("恐慌值_60日", panic_l)  # v2.1: 混合恐慌
    chase = df["追涨狂热"]; disp = df["disp_ma250"]
    settle = df["settle"]; oversold = df["oversold_local"]
    ps = df["买入预估成功率_5d"]
    sig = pd.Series("", index=df.index)
    buy_score = pd.Series(np.nan, index=df.index)

    # ---- v2.1: 混合恐慌 + 深度回调放宽 ----
    panic_mix = panic_l * 0.5 + panic_60 * 0.5  # 混合恐慌值
    deep_pullback = df["pullback_from_high"] < -0.15  # 回调>15%

    # ---- 买入A：恐慌企稳 + v2增强 ----
    # v2.1: 恐慌改用混合值；深度回调时settle放宽（不要求涨，只要不继续放量跌）
    base_a = (panic_mix >= p["buy_panic_l"]) & (risk_l < p["buy_risk_l"]) & (mkt < p["buy_mkt"]) & (ps >= p["buy_prob"])
    # settle条件：普通情况要求settle=1，深度回调时只要求不是放量下跌
    settle_ok = (settle == 1) | (deep_pullback & (df["heavy_down_5d"] < 0.6))
    base_a = base_a & settle_ok

    # v2.1: pullback_max 修复：默认至少回调2%，除非是深度回调场景
    pb_ok = (df["pullback_from_high"] < -0.02) | deep_pullback
    enhance_a = (
        (df["vol_quality"] > p["vol_quality_min"]) &
        pb_ok &
        (df["heavy_down_5d"] < p["heavy_down_max"])
    )
    div_bull = (df["macd_bull_div"] == 1) | (df["vol_bull_div"] == 1)
    cond_a = base_a & enhance_a

    # ---- 买入B：超卖缩量 + v2增强 ----
    base_b = (oversold >= p["oversold"]) & (df["vol_shrink"] == 1) & (risk < p["buy_risk_g"]) & (mkt < p["buy_mkt"]) & (ps >= p["buy_prob"])
    enhance_b = (
        (df["vol_quality"] > p["vol_quality_min"]) &
        pb_ok  # v2.1: 同样要求回调至少2%
    )
    cond_b = base_b & enhance_b

    # ---- 买入C：底背离买入 ----
    cond_c = (
        (div_bull) &
        (risk < p["buy_risk_g"] + 10) &
        (mkt < p["buy_mkt"]) &
        (ps >= p["buy_prob"] - 5) &
        (df["vol_shrink_streak"] >= p["shrink_streak"])
    )

    # ---- v2.1: 无硬过滤，直接评分 ----
    buy_idx = df.index[(cond_a | cond_b | cond_c)]
    for i in buy_idx:
        sig.iloc[i] = "买入"
        buy_score.iloc[i] = score_buy_point(df.iloc[i])

    # 长线买入
    lt_idx = df.index[(disp <= p["lt_disp"]) & (risk < 50) & (mkt < 80) & (ps >= p["lt_prob"])]
    for i in lt_idx:
        if sig.iloc[i] == "":
            sig.iloc[i] = "长线买入"
            buy_score.iloc[i] = score_buy_point(df.iloc[i])

    # ---- 卖出信号（v2增强） ----
    base_sell = (chase >= p["sell_chase"]) & (risk >= p["sell_risk"])
    div_bear = (df["macd_bear_div"] == 1) | (df["vol_bear_div"] == 1)
    sell = base_sell | (div_bear & (risk >= p["sell_risk"] - 5))
    sig[sell] = "卖出"

    df["signal"] = sig
    df["buy_score"] = buy_score

    # ---- 评分归一化 ----
    raw_scores = buy_score.dropna()
    if len(raw_scores) > 10:
        rmin, rmax = raw_scores.min(), raw_scores.max()
        if rmax > rmin:
            df["buy_score"] = (buy_score - rmin) / (rmax - rmin) * 9 + 1
            df["buy_score"] = df["buy_score"].round(1)
        else:
            df["buy_score"] = buy_score.fillna(0).round(1)
    else:
        df["buy_score"] = buy_score.fillna(0).round(1)

    return df

def backtest(df, p):
    close = df["close"].values; ret = df["ret"].fillna(0).values
    sig = df["signal"].values; n = len(df)
    pos = np.zeros(n, int); trades = []
    holding = False; ei = 0; last_buy = -999; last_sell = -999; peak = 0
    sp = p["stop"]; tr = p["trail"]; space = p["space"]; tstop = p["tstop"]

    for i in range(n):
        if not holding:
            if sig[i] in ("买入", "长线买入") and (i - last_buy) >= space:
                holding = True; ei = i; last_buy = i; peak = close[i]; pos[i] = 1
        else:
            pos[i] = 1; peak = max(peak, close[i])
            exit_now = False; reason = ""
            if sig[i] == "卖出" and (i - last_sell) >= space:
                exit_now = True; reason = "卖出信号"; last_sell = i
            elif close[i] / close[ei] - 1 <= -sp:
                exit_now = True; reason = "硬止损"
            elif peak > close[ei] * 1.01 and close[i] / peak - 1 <= -tr:
                exit_now = True; reason = "移动止盈"
            elif (i - ei) >= tstop:
                exit_now = True; reason = "时间止损"
            if exit_now:
                pnl = close[i] / close[ei] - 1
                trades.append({
                    "entry_i": ei, "exit_i": i,
                    "entry_date": str(df["date"].iloc[ei])[:10],
                    "exit_date":  str(df["date"].iloc[i])[:10],
                    "entry_close": close[ei], "exit_close": close[i],
                    "pnl": pnl, "hold_days": i - ei, "exit_reason": reason})
                holding = False; pos[i] = 0
    if holding:
        i = n - 1; pnl = close[i] / close[ei] - 1
        trades.append({
            "entry_i": ei, "exit_i": i,
            "entry_date": str(df["date"].iloc[ei])[:10],
            "exit_date":  str(df["date"].iloc[i])[:10],
            "entry_close": close[ei], "exit_close": close[i],
            "pnl": pnl, "hold_days": i - ei, "exit_reason": "期末平仓"})
    eq = np.cumprod(1 + ret * pos); bh = np.cumprod(1 + ret)
    return trades, eq, bh, pos

def fwd_acc(df, sig_type, horizon):
    idx = df.index[df["signal"].str.contains("买入|长线" if "买入" in sig_type else sig_type, na=False)].tolist()
    ok = tot = 0
    for i in idx:
        j = i + horizon
        if j < len(df):
            r = df["close"].iloc[j] / df["close"].iloc[i] - 1; tot += 1
            if "买入" in sig_type and r > 0: ok += 1
            if sig_type == "卖出" and r < 0: ok += 1
    return round(ok / tot * 100, 1) if tot else None, tot

def stats(trades, eq, bh, ret, df):
    n = len(ret); total = eq[-1] - 1; bh_total = bh[-1] - 1
    ann = (1 + total) ** (252 / n) - 1 if n > 0 else 0
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    wins = [t for t in trades if t["pnl"] > 0]
    wr = len(wins) / len(trades) if trades else 0
    avg_pnl = np.mean([t["pnl"] for t in trades]) if trades else 0
    nb = len(df.index[df["signal"].str.contains("买入|长线", na=False)])
    ns = len(df.index[df["signal"] == "卖出"])
    a5, _  = fwd_acc(df, "买入", 5)
    a7, _  = fwd_acc(df, "买入", 7)
    a14, _ = fwd_acc(df, "买入", 14)
    return {
        "trades": len(trades), "wins": len(wins),
        "win_rate": round(wr * 100, 1),
        "avg_pnl": round(avg_pnl * 100, 2),
        "total_return": round(total * 100, 1),
        "annual_return": round(ann * 100, 1),
        "max_dd": round(mdd * 100, 1),
        "buyhold_return": round(bh_total * 100, 1),
        "excess": round((total - bh_total) * 100, 1),
        "avg_hold": round(np.mean([t["hold_days"] for t in trades]), 1) if trades else 0,
        "n_buy": nb, "n_sell": ns,
        "buy_acc5": a5, "buy_acc7": a7, "buy_acc14": a14,
    }

def score(s):
    a7 = s["buy_acc7"] if s["buy_acc7"] is not None else 50
    return (a7 / 100 * 0.35
          + max(s["total_return"], -250) / 250 * 0.20
          + (-s["max_dd"]) / 100 * 0.15
          + max(s["excess"], -200) / 200 * 0.10
          + max(s["win_rate"] / 100, 0) * 0.10
          + max(s["trades"] / 40, 0) * 0.10)

def run_grid(sectors, grid):
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    best_score = -1e9; best_params = None
    total = len(combos)
    for idx, cp in enumerate(combos):
        sc = 0.0
        for name, d in sectors.items():
            dd = gen_signals(d, cp); tr, eq, bh, pos = backtest(dd, cp)
            sc += score(stats(tr, eq, bh, dd["ret"].values, dd))
        sc /= len(sectors)
        if sc > best_score: best_score = sc; best_params = cp
        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx+1}/{total}, 当前最优评分={best_score:.4f}")
    return best_params, best_score

def _dim_label(name, score):
    """为评分维度生成简短标签"""
    labels = {
        "回调深度":   {10:"暴跌40%+", 9:"跌30%+", 8:"跌25%+", 7:"跌20%+", 5.5:"跌15%+", 4.5:"跌12%+", 3:"跌8%+", 2:"跌5%+", 1.5:"跌3%+", 1:"几乎没跌"},
        "背离确认":   {10:"MACD+量价双底背离", 8:"MACD底背离", 6.5:"量价底背离", 4.5:"MACD金叉", 3:"DIF转正", 1.5:"无背离"},
        "量价健康度": {10:"极度健康", 8:"很健康", 6:"正常", 4:"偏弱", 2:"差", 1:"极差"},
        "趋势健康度": {10:"多头排列", 8:"偏多", 6:"震荡偏多", 5:"中性", 4:"偏空震荡", 3:"空头排列", 2:"全空头+连跌", 1:"全空头加速跌"},
        "支撑位强度": {10:"双支撑", 7.5:"强支撑", 5:"均线附近", 3.5:"弱支撑", 2:"无支撑", 1:"跌破所有均线"},
        "恐慌极端度": {10:"极度恐慌(>85)", 8:"高度恐慌(>70)", 6:"中度恐慌(>55)", 4:"轻度恐慌(>40)", 2.5:"低恐慌(>25)", 1.5:"无恐慌"},
        "企稳确认度": {10:"缩量4天+企稳", 8:"缩量3天企稳", 6:"缩量2天企稳", 4:"企稳", 2:"未企稳"},
    }
    info = labels.get(name, {})
    best = min(info.keys(), key=lambda k: abs(k - score))
    return info.get(best, "")

def reason(row, sig_type):
    d = str(row["date"])[:10]

    if sig_type == "长线买入":
        disp = row["disp_ma250"] * 100
        ps = row["买入预估成功率_5d"]
        bscore = row.get("buy_score", None)
        total_str = f"{bscore:.1f}" if (bscore is not None and not np.isnan(bscore)) else "—"
        return f"{d} 长线买入：较年线低{abs(disp):.1f}%，预估5日成功率={ps}%。 [综合{total_str}分]"

    if sig_type == "买入":
        _, dims = score_buy_point(row, return_detail=True)
        bscore = row.get("buy_score", None)
        total_str = f"{bscore:.1f}" if (bscore is not None and not np.isnan(bscore)) else "—"
        order = ["回调深度", "背离确认", "量价健康度", "趋势健康度", "支撑位强度", "恐慌极端度", "企稳确认度"]
        parts = []
        for dim_name in order:
            s = dims.get(dim_name)
            if s is None:
                continue
            label = _dim_label(dim_name, s)
            parts.append(f"{dim_name}：{s:.0f}分（{label}）" if s == int(s) else f"{dim_name}：{s:.1f}分（{label}）")
        dim_str = " → ".join(parts)
        return f"{d} 买入 [综合{total_str}分]\n  {dim_str}"

    if sig_type == "卖出":
        risk = row["板块风险"]
        chase = row["追涨狂热"]
        macd_bear = row.get("macd_bear_div", 0)
        vol_bear = row.get("vol_bear_div", 0)
        extras = []
        if macd_bear: extras.append("MACD顶背离")
        if vol_bear: extras.append("量价顶背离")
        extra_str = "，".join(extras)
        base = f"{d} 卖出：追涨狂热={chase}，板块风险={risk}"
        if extra_str:
            base += f"，{extra_str}"
        return base + "。"

    return ""

# ============ 默认参数（v2） ============
DEFAULT_PARAMS_V2 = dict(
    # v1 保留参数
    lt_disp=-0.20, lt_prob=43,
    buy_panic_l=58, buy_risk_l=67,
    buy_mkt=74, buy_prob=43, buy_risk_g=50,
    sell_chase=74, sell_risk=74, oversold=0.50,
    stop=0.04, trail=0.025, space=3, tstop=22,
    # v2 新增参数
    vol_quality_min=-0.5,    # 量价质量最低值（-2~2，负=畸形）
    pullback_max=-0.02,      # 回调幅度上限（必须回调至少2%）
    heavy_down_max=0.4,      # 近5日放量下跌天数占比上限
    shrink_streak=2,         # 连续缩量天数（买入C用）
)

# ============ 快速模式 ============
def quick_run():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 加载数据…")
    mkt = pd.read_csv(f"{DATA_DIR}/{MARKET_FILE}", parse_dates=["date"])

    params_path = f"{OUT_DIR}/swing_v2_params.json"
    if os.path.exists(params_path):
        bp = json.load(open(params_path, encoding="utf-8"))
        params = bp["params"]
        print(f"  加载v2参数，评分={bp.get('score', 'N/A')}")
    else:
        params = DEFAULT_PARAMS_V2
        print(f"  使用默认v2参数")

    print(f"[{time.strftime('%H:%M:%S')}] 计算v2指标与买入成功率…")
    sectors = {}
    for name in SECTORS:
        d = pd.read_csv(f"{DATA_DIR}/sector_{name}.csv", parse_dates=["date"])
        d = add_indicators(d)
        d = add_market_risk(d, mkt)
        d = add_buy_success_prob_5d(d)
        sectors[name] = d
    print(f"  指标计算耗时: {time.time()-t0:.0f}s")

    print(f"[{time.strftime('%H:%M:%S')}] 生成v2信号与回测…")
    final_out = {}
    for name, d in sectors.items():
        dd = gen_signals(d, params)
        tr, eq, bh, pos = backtest(dd, params)
        final_out[name] = stats(tr, eq, bh, dd["ret"].values, dd)
        dd["reason"] = [reason(r, sg) for r, sg in zip(dd.to_dict("records"), dd["signal"].fillna(""))]
        dd.to_csv(f"{OUT_DIR}/swing_v2_{name}_signal.csv", index=False, encoding="utf-8-sig")

    with open(f"{OUT_DIR}/swing_v2_stats.json", "w", encoding="utf-8") as f:
        json.dump(final_out, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'板块':6s} {'交易':>4s} {'胜率':>6s} {'收益':>7s} {'回撤':>6s} {'超额':>7s} {'持仓':>5s} {'7日准确':>7s}")
    for k, v in final_out.items():
        a7 = f"{v['buy_acc7']}%" if v['buy_acc7'] is not None else "—"
        print(f"{k:6s} {v['trades']:4d} {v['win_rate']:5.1f}% {v['total_return']:6.1f}% "
              f"{v['max_dd']:5.1f}% {v['excess']:6.1f}% {v['avg_hold']:4.1f}日 "
              f"{a7:>7s}")

    print(f"\n{'='*60}")
    print(f"今日信号 ({pd.Timestamp.today().strftime('%Y-%m-%d')})：")
    for name, d in sectors.items():
        today_row = d[d["date"] == d["date"].max()]
        if len(today_row) > 0:
            r = today_row.iloc[-1]
            sig = r.get("signal", "")
            ps = r.get("买入预估成功率_5d", "—")
            if sig and sig != "":
                print(f"  ⚡ {name}: {sig} | 预估成功率={ps}% | {r.get('reason','')[:100]}")
            else:
                print(f"     {name}: 无信号 | 预估成功率={ps}% | 板块风险={r.get('板块风险','—')}")

    print(f"\n总耗时: {time.time()-t0:.0f}s")
    return sectors, final_out, params

# ============ 优化模式 ============
def optimize_run():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 加载大盘基准…")
    mkt = pd.read_csv(f"{DATA_DIR}/{MARKET_FILE}", parse_dates=["date"])

    print(f"[{time.strftime('%H:%M:%S')}] 计算v2指标与预估成功率…")
    sectors = {}
    for name in SECTORS:
        d = pd.read_csv(f"{DATA_DIR}/sector_{name}.csv", parse_dates=["date"])
        d = add_indicators(d); d = add_market_risk(d, mkt); d = add_buy_success_prob_5d(d)
        sectors[name] = d
    print(f"  指标计算耗时: {time.time()-t0:.0f}s")

    # 基线
    base = DEFAULT_PARAMS_V2
    print(f"[{time.strftime('%H:%M:%S')}] v2基线回测…")
    for name, d in sectors.items():
        dd = gen_signals(d, base); tr, eq, bh, pos = backtest(dd, base)
        sc = score(stats(tr, eq, bh, dd["ret"].values, dd))
        print(f"  {name}: 评分={sc:.4f}")

    # 迭代1：v1参数在已知最优附近微调 + v2新参数粗搜
    # v1最优: lt_disp=-0.20, lt_prob=43, buy_panic_l=58, buy_risk_l=67, buy_mkt=74,
    #          buy_prob=43, buy_risk_g=50, sell_chase=74, sell_risk=74, oversold=0.50,
    #          stop=0.04, trail=0.025, space=3, tstop=22
    grid1 = dict(
        lt_disp=[-0.22, -0.20], lt_prob=[43],
        buy_panic_l=[55, 60], buy_risk_l=[60, 67],
        buy_mkt=[74], buy_prob=[40, 45], buy_risk_g=[48, 52],
        sell_chase=[72, 76], sell_risk=[72, 76],
        oversold=[0.45, 0.55],
        stop=[0.04], trail=[0.025], space=[3], tstop=[22],
        # v2 新参数（各2-3个值）
        vol_quality_min=[-0.5, -0.2],
        pullback_max=[-0.02, -0.04],
        heavy_down_max=[0.3, 0.5],
        shrink_streak=[1, 2],
    )
    n1 = np.prod([len(grid1[k]) for k in grid1])
    print(f"[{time.strftime('%H:%M:%S')}] 迭代1 网格搜索（{n1} 组合）…")
    bp1, bs1 = run_grid(sectors, grid1)
    print(f"  迭代1 评分={bs1:.4f}")

    # 迭代2：局部细化
    grid2 = {}
    for k in grid1:
        if k in ("stop", "trail", "space", "tstop"):
            grid2[k] = [bp1[k]]
        else:
            step = grid1[k][1] - grid1[k][0] if len(grid1[k]) >= 2 else grid1[k][0] * 0.2
            grid2[k] = sorted(set([max(bp1[k] - step, grid1[k][0]), bp1[k], min(bp1[k] + step, grid1[k][-1])]))
    n2 = np.prod([len(grid2[k]) for k in grid2])
    print(f"[{time.strftime('%H:%M:%S')}] 迭代2 局部细化（{n2} 组合）…")
    bp2, bs2 = run_grid(sectors, grid2)
    print(f"  迭代2 评分={bs2:.4f}")

    best_params, best_score = (bp2, bs2) if bs2 >= bs1 else (bp1, bs1)

    print(f"[{time.strftime('%H:%M:%S')}] 生成最终信号与统计…")
    final_out = {}
    for name, d in sectors.items():
        dd = gen_signals(d, best_params); tr, eq, bh, pos = backtest(dd, best_params)
        final_out[name] = stats(tr, eq, bh, dd["ret"].values, dd)
        dd["reason"] = [reason(r, sg) for r, sg in zip(dd.to_dict("records"), dd["signal"].fillna(""))]
        dd.to_csv(f"{OUT_DIR}/swing_v2_{name}_signal.csv", index=False, encoding="utf-8-sig")

    with open(f"{OUT_DIR}/swing_v2_stats.json", "w", encoding="utf-8") as f:
        json.dump(final_out, f, ensure_ascii=False, indent=2, default=str)
    with open(f"{OUT_DIR}/swing_v2_params.json", "w", encoding="utf-8") as f:
        json.dump({"params": best_params, "score": round(best_score, 4),
                   "iter1": {"params": bp1, "score": round(bs1, 4)},
                   "iter2": {"params": bp2, "score": round(bs2, 4)}}, f, ensure_ascii=False, indent=2)

    print(f"\n{'板块':6s} {'交易':>4s} {'胜率':>6s} {'收益':>7s} {'回撤':>6s} {'超额':>7s} {'持仓':>5s} {'7日准确':>7s}")
    for k, v in final_out.items():
        a7 = f"{v['buy_acc7']}%" if v['buy_acc7'] is not None else "—"
        print(f"{k:6s} {v['trades']:4d} {v['win_rate']:5.1f}% {v['total_return']:6.1f}% "
              f"{v['max_dd']:5.1f}% {v['excess']:6.1f}% {v['avg_hold']:4.1f}日 "
              f"{a7:>7s}")

    print(f"\n总耗时: {time.time()-t0:.0f}s | 最终评分: {best_score:.4f}")
    return sectors, final_out, best_params

# ============ 主入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="执行两轮网格搜索优化参数")
    args = parser.parse_args()

    if args.optimize:
        print("模式：v2参数优化（网格搜索，含新指标）")
        optimize_run()
    else:
        print("模式：v2快速信号生成（固定参数）")
        quick_run()
