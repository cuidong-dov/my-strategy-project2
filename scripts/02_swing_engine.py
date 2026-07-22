#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股板块评估系统 · 小波段策略引擎
核心：局部恐慌+缩量企稳 → 5日方向标定 → 两轮网格迭代 → 输出信号CSV与统计JSON
"""
import pandas as pd, numpy as np, json, os, itertools, statistics as st, time, sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

SECTORS = ["半导体","消费电子","通信","电池","电力","煤炭"]
MARKET_FILE = "index_沪深300.csv"

def pct_rank(s, w=250):
    return s.rolling(w, min_periods=max(20, w//4)).apply(
        lambda x: (x[-1] <= x).mean() * 100.0, raw=True)

def add_indicators(df):
    df = df.copy().sort_values("date").reset_index(drop=True)
    c = df["close"]; v = df["volume"]
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

    # ---- 五大字段（全局250日分位） ----
    df["板块风险"] = pct_rank(0.6 * df["disp_ma250"] + 0.4 * df["vol_z"], 250).round(1)
    weak = (df["disp_ma20"].clip(upper=0) / -0.15).clip(0, 1)
    df["抄底狂热"] = pct_rank(df["vol_ratio"] * (0.3 + 0.7 * weak), 250).round(1)
    strong = (df["disp_ma20"].clip(lower=0) / 0.15).clip(0, 1)
    df["追涨狂热"] = pct_rank(df["vol_ratio"] * (0.3 + 0.7 * strong), 250).round(1)
    panic_raw = ((df["dd_20"].clip(upper=0) / -0.20).clip(0, 1) * 0.6
               + (df["vol_ratio"].clip(upper=3) - 1).clip(0, 2) / 2 * 0.2
               + (df["ret"].clip(upper=0) / -0.04).clip(0, 1) * 0.2)
    df["恐慌值"] = pct_rank(panic_raw, 250).round(1)

    # ---- 小波段专用 ----
    df["恐慌值_局部"]  = pct_rank(panic_raw, 20).round(1)
    df["板块风险_局部"] = pct_rank(0.6 * df["disp_ma250"] + 0.4 * df["vol_z"], 20).round(1)
    df["close_prev"]  = c.shift(1)
    df["price_up"]    = (c > df["close_prev"]).astype(int)
    df["vol_shrink"]  = (df["vol_ratio"] < 1.0).astype(int)
    df["settle"]      = ((df["price_up"] == 1) & (df["vol_shrink"] == 1)).astype(int)
    df["oversold_local"] = ((-df["dd_5"]).clip(0, 0.10) / 0.10).clip(0, 1)
    return df

def add_market_risk(df, mkt):
    m = add_indicators(mkt)
    mr = m[["date", "板块风险"]].rename(columns={"板块风险": "大盘风险"})
    df = df.merge(mr, on="date", how="left")
    df["大盘风险"] = df["大盘风险"].ffill().bfill()
    return df

def add_buy_success_prob_5d(df):
    """分箱法：五因子 → 历史同条件组合下5日后涨跌概率"""
    n = len(df); prob = np.full(n, 50.0)
    close = df["close"].values
    feats = np.column_stack([
        df["恐慌值_局部"].values, df["板块风险_局部"].values,
        df["抄底狂热"].values,    df["vol_ratio"].values,
        (df["disp_ma5"] * 100).values])
    bins = 5
    for i in range(120, n):
        w = feats[i-120:i]
        fwd_ret = np.array([1 if close[j+5] > close[j] else 0 for j in range(i-120, i-5)])
        w = w[:len(fwd_ret)]
        if len(w) < 30: continue
        bin_ids = []
        for k in range(5):
            edges = np.percentile(w[:, k], np.linspace(0, 100, bins + 1))
            edges = np.unique(np.round(edges, 1))
            if len(edges) < 3: bin_ids.append(np.zeros(len(w), dtype=int)); continue
            ids = np.digitize(w[:, k], edges[1:-1])
            bin_ids.append(np.clip(ids, 0, bins - 1))
        bin_ids = np.column_stack(bin_ids)
        cur_bin = []
        for k in range(5):
            edges = np.percentile(w[:, k], np.linspace(0, 100, bins + 1))
            edges = np.unique(np.round(edges, 1))
            if len(edges) < 3: cur_bin.append(0); continue
            cid = np.digitize([feats[i, k]], edges[1:-1])[0]
            cur_bin.append(np.clip(cid, 0, bins - 1))
        cur_bin = tuple(cur_bin)
        match = np.ones(len(w), dtype=bool)
        for k in range(5): match = match & (bin_ids[:, k] == cur_bin[k])
        if match.sum() >= 3:
            prob[i] = round(fwd_ret[match].mean() * 100, 1)
    df["买入预估成功率_5d"] = prob
    df["买入预估成功率_5d"] = df["买入预估成功率_5d"].ffill().bfill().clip(0, 100).round(1)
    return df

def gen_signals(df, p):
    df = df.copy()
    risk = df["板块风险"]; mkt = df["大盘风险"]
    risk_l = df["板块风险_局部"]; panic_l = df["恐慌值_局部"]
    chase = df["追涨狂热"]; disp = df["disp_ma250"]
    settle = df["settle"]; oversold = df["oversold_local"]
    ps = df["买入预估成功率_5d"]
    sig = pd.Series("", index=df.index)

    cond_a = (panic_l >= p["buy_panic_l"]) & (settle == 1) & (risk_l < p["buy_risk_l"]) & (mkt < p["buy_mkt"]) & (ps >= p["buy_prob"])
    cond_b = (oversold >= p["oversold"]) & (df["vol_shrink"] == 1) & (risk < p["buy_risk_g"]) & (mkt < p["buy_mkt"]) & (ps >= p["buy_prob"])
    buy = cond_a | cond_b
    lt = (disp <= p["lt_disp"]) & (risk < 50) & (mkt < 80) & (ps >= p["lt_prob"])
    sell = (chase >= p["sell_chase"]) & (risk >= p["sell_risk"])

    sig[buy] = "买入"; sig[lt] = "长线买入"; sig[sell] = "卖出"
    df["signal"] = sig
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
    idx = df.index[df["signal"] == sig_type].tolist()
    ok = tot = 0
    for i in idx:
        j = i + horizon
        if j < len(df):
            r = df["close"].iloc[j] / df["close"].iloc[i] - 1; tot += 1
            if sig_type in ("买入", "长线买入") and r > 0: ok += 1
            if sig_type == "卖出" and r < 0: ok += 1
    return round(ok / tot * 100, 1) if tot else None, tot

def stats(trades, eq, bh, ret, df):
    n = len(ret); total = eq[-1] - 1; bh_total = bh[-1] - 1
    ann = (1 + total) ** (252 / n) - 1 if n > 0 else 0
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    wins = [t for t in trades if t["pnl"] > 0]
    wr = len(wins) / len(trades) if trades else 0
    avg_pnl = np.mean([t["pnl"] for t in trades]) if trades else 0
    nb = len(df.index[df["signal"].isin(["买入", "长线买入"])])
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
    for cp in combos:
        sc = 0.0
        for name, d in sectors.items():
            dd = gen_signals(d, cp); tr, eq, bh, pos = backtest(dd, cp)
            sc += score(stats(tr, eq, bh, dd["ret"].values, dd))
        sc /= len(sectors)
        if sc > best_score: best_score = sc; best_params = cp
    return best_params, best_score

def reason(row, sig_type):
    risk = row["板块风险"]; mkt = row["大盘风险"]; dip = row["抄底狂热"]
    chase = row["追涨狂热"]; panic = row["恐慌值"]; panic_l = row["恐慌值_局部"]
    ps = row["买入预估成功率_5d"]; d = str(row["date"])[:10]
    disp = row["disp_ma250"] * 100
    if sig_type == "长线买入":
        return f"{d} 小波段长线：较年线低{abs(disp):.1f}%，预估5日成功率={ps}%。"
    if sig_type == "买入":
        return f"{d} 小波段买入：局部恐慌={panic_l}，板块风险={risk}，大盘风险={mkt}，预估5日成功率={ps}%，缩量企稳信号。"
    if sig_type == "卖出":
        return f"{d} 小波段卖出：追涨狂热={chase}，板块风险={risk}，亢奋止盈。"
    return ""

# ============ 主流程 ============
def run():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 加载大盘基准…")
    mkt = pd.read_csv(f"{DATA_DIR}/{MARKET_FILE}", parse_dates=["date"])

    print(f"[{time.strftime('%H:%M:%S')}] 计算指标与预估成功率…")
    sectors = {}
    for name in SECTORS:
        d = pd.read_csv(f"{DATA_DIR}/sector_{name}.csv", parse_dates=["date"])
        d = add_indicators(d); d = add_market_risk(d, mkt); d = add_buy_success_prob_5d(d)
        sectors[name] = d
    print(f"  指标计算耗时: {time.time()-t0:.0f}s")

    # 基线参数
    base = dict(lt_disp=-0.20, lt_prob=48, buy_panic_l=65, buy_risk_l=60,
                buy_mkt=72, buy_prob=48, buy_risk_g=55, oversold=0.6,
                sell_chase=70, sell_risk=72, sell_disp=0.12,
                stop=0.04, trail=0.025, space=4, tstop=20)

    print(f"[{time.strftime('%H:%M:%S')}] 基线回测…")
    base_out = {}
    for name, d in sectors.items():
        dd = gen_signals(d, base); tr, eq, bh, pos = backtest(dd, base)
        base_out[name] = stats(tr, eq, bh, dd["ret"].values, dd)

    # 迭代1
    grid1 = dict(lt_disp=[-0.22, -0.20], lt_prob=[45, 50],
                 buy_panic_l=[60, 65, 70], buy_risk_l=[55, 60, 65],
                 buy_mkt=[70, 74], buy_prob=[45, 48, 52], buy_risk_g=[52, 56],
                 sell_chase=[68, 72], sell_risk=[70, 74], sell_disp=[0.12],
                 oversold=[0.55, 0.65],
                 stop=[0.04], trail=[0.025, 0.03], space=[3, 5], tstop=[18, 22])
    print(f"[{time.strftime('%H:%M:%S')}] 迭代1 网格搜索（{np.prod([len(grid1[k]) for k in grid1])} 组合）…")
    bp1, bs1 = run_grid(sectors, grid1)
    print(f"  迭代1 评分={bs1:.4f} 参数={bp1}")

    # 迭代2
    grid2 = dict(lt_disp=[bp1["lt_disp"]], lt_prob=[bp1["lt_prob"]-2, bp1["lt_prob"], bp1["lt_prob"]+2],
                 buy_panic_l=[bp1["buy_panic_l"]-2, bp1["buy_panic_l"], bp1["buy_panic_l"]+2],
                 buy_risk_l=[bp1["buy_risk_l"]-2, bp1["buy_risk_l"], bp1["buy_risk_l"]+2],
                 buy_mkt=[bp1["buy_mkt"]], buy_prob=[bp1["buy_prob"]-2, bp1["buy_prob"], bp1["buy_prob"]+2],
                 buy_risk_g=[bp1["buy_risk_g"]-2, bp1["buy_risk_g"], bp1["buy_risk_g"]+2],
                 sell_chase=[bp1["sell_chase"]-2, bp1["sell_chase"], bp1["sell_chase"]+2],
                 sell_risk=[bp1["sell_risk"]-2, bp1["sell_risk"], bp1["sell_risk"]+2],
                 sell_disp=[bp1["sell_disp"]], oversold=[bp1["oversold"]-0.05, bp1["oversold"], bp1["oversold"]+0.05],
                 stop=[bp1["stop"]], trail=[bp1["trail"]], space=[bp1["space"]], tstop=[bp1["tstop"]])
    print(f"[{time.strftime('%H:%M:%S')}] 迭代2 局部细化…")
    bp2, bs2 = run_grid(sectors, grid2)
    print(f"  迭代2 评分={bs2:.4f} 参数={bp2}")
    best_params, best_score = (bp2, bs2) if bs2 >= bs1 else (bp1, bs1)

    # 最终输出
    print(f"[{time.strftime('%H:%M:%S')}] 生成最终信号与统计…")
    final_out = {}
    for name, d in sectors.items():
        dd = gen_signals(d, best_params); tr, eq, bh, pos = backtest(dd, best_params)
        final_out[name] = stats(tr, eq, bh, dd["ret"].values, dd)
        dd["reason"] = [reason(r, sg) for r, sg in zip(dd.to_dict("records"), dd["signal"].fillna(""))]
        dd.to_csv(f"{OUT_DIR}/swing_{name}_signal.csv", index=False, encoding="utf-8-sig")

    with open(f"{OUT_DIR}/swing_stats.json", "w", encoding="utf-8") as f:
        json.dump(final_out, f, ensure_ascii=False, indent=2, default=str)
    with open(f"{OUT_DIR}/swing_params.json", "w", encoding="utf-8") as f:
        json.dump({"params": best_params, "score": round(best_score, 4),
                   "iter1": {"params": bp1, "score": round(bs1, 4)},
                   "iter2": {"params": bp2, "score": round(bs2, 4)}}, f, ensure_ascii=False, indent=2)

    # 汇总打印
    print(f"\n{'板块':6s} {'交易':>4s} {'胜率':>6s} {'收益':>7s} {'回撤':>6s} {'超额':>7s} {'持仓':>5s} {'5日准确':>7s} {'7日准确':>7s}")
    for k, v in final_out.items():
        print(f"{k:6s} {v['trades']:4d} {v['win_rate']:5.1f}% {v['total_return']:6.1f}% "
              f"{v['max_dd']:5.1f}% {v['excess']:6.1f}% {v['avg_hold']:4.1f}日 "
              f"{v['buy_acc5'] or '—':>7s} {v['buy_acc7'] or '—':>7s}")

    print(f"\n总耗时: {time.time()-t0:.0f}s | 最终评分: {best_score:.4f}")
    return sectors, final_out, best_params

if __name__ == "__main__":
    run()
