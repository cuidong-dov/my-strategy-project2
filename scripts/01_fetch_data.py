#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股板块评估系统 · 数据拉取模块
拉取 6 个板块(同花顺行业指数) + 大盘基准(新浪指数) 2021-01-01 ~ 今天的日线数据。
支持实时数据fallback：当天日K线未出时，用同花顺实时行情填充，后续日线出来后自动修正。
"""
import akshare as ak, pandas as pd, time, os, sys, json, re
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
END = pd.Timestamp.today().strftime("%Y%m%d")
START = "20210101"
TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")

# 板块名称 → (显示名, THS行业名, THS代码)
SECTORS = {
    "半导体":   ("半导体",       "半导体",       "881121"),
    "消费电子": ("消费电子",     "消费电子",     "881124"),
    "通信":     ("通信",         "通信设备",     "881129"),
    "电池":     ("电池",         "电池",         "881281"),
    "电力":     ("电力",         "电力",         "881145"),
    "煤炭":     ("煤炭",         "煤炭开采加工", "881105"),
}

def retry(fn, n=4, wait=3):
    last = None
    for i in range(n):
        try: return fn()
        except Exception as e:
            last = e
            print(f"  重试{i+1}/{n}: {repr(e)[:100]}", file=sys.stderr)
            time.sleep(wait)
    raise last

def fetch_sector(ths_name):
    """通过akshare拉取同花顺行业板块日线数据"""
    df = retry(lambda: ak.stock_board_industry_index_ths(
        symbol=ths_name, start_date=START, end_date=END))
    df = df.rename(columns={"日期":"date","开盘价":"open","最高价":"high",
                            "最低价":"low","收盘价":"close","成交量":"volume","成交额":"amount"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df["is_realtime"] = 0
    return df[df["date"] >= "2021-01-01"]

def fetch_index(code):
    """通过akshare拉取大盘指数日线数据"""
    df = retry(lambda: ak.stock_zh_index_daily(symbol=code))
    df = df.rename(columns={"date":"date","open":"open","high":"high",
                            "low":"low","close":"close","volume":"volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df["amount"] = float("nan")
    return df[df["date"] >= "2021-01-01"]

def fetch_realtime_sector(ths_code):
    """
    从同花顺实时行情接口获取当天实时数据。
    返回 dict: {date, open, high, low, close, volume, amount, is_realtime=1}
    若失败返回 None。
    """
    url = f"https://d.10jqka.com.cn/v2/realhead/bk_{ths_code}/last.js"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.10jqka.com.cn/"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"    实时API返回状态码 {r.status_code}", file=sys.stderr)
            return None
        text = re.sub(r'^quotebridge_v2_realhead_bk_\d+_last\(', '', r.text).rstrip(')')
        data = json.loads(text)
        items = data["items"]

        # 解析时间，提取日期
        rt_time = items.get("time", "")
        rt_date_match = re.match(r'(\d{4}-\d{2}-\d{2})', rt_time)
        if not rt_date_match:
            print(f"    实时API时间格式异常: {rt_time}", file=sys.stderr)
            return None
        rt_date = rt_date_match.group(1)

        close  = float(items.get("7", 0))     # 最新价
        high   = float(items.get("8", 0))     # 最高价
        low    = float(items.get("9", 0))     # 最低价
        open_p = float(items.get("10", 0))    # 开盘价
        pre_close = float(items.get("6", 0))  # 昨收
        amount = float(items.get("13", 0))    # 成交额（元）
        volume = float(items.get("14", 0))    # 成交量（手）— 备选字段

        # 验证数据有效性
        if close <= 0 or open_p <= 0:
            print(f"    实时API返回数据无效: close={close}, open={open_p}", file=sys.stderr)
            return None

        return {
            "date": rt_date,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "is_realtime": 1
        }
    except Exception as e:
        print(f"    实时API异常: {repr(e)[:100]}", file=sys.stderr)
        return None

def merge_realtime(df, ths_code, sector_name):
    """
    检查日线数据是否缺少今天的数据，如果缺少则用实时API补充。
    同时清理之前标记为实时数据、现在日线已有的行（自动修正）。
    """
    # 1. 清理旧的实时数据：如果某天的 is_realtime=1，但日线数据中也有该天，删除实时行
    realtime_rows = df[df["is_realtime"] == 1]
    for _, rr in realtime_rows.iterrows():
        rd = rr["date"]
        if len(df[(df["date"] == rd) & (df["is_realtime"] == 0)]) > 0:
            df = df[~((df["date"] == rd) & (df["is_realtime"] == 1))]
            print(f"  [修正] {sector_name} {str(rd)[:10]} 日线已覆盖实时数据，自动修正。")

    # 2. 检查今天的数据是否缺失
    today_ts = pd.Timestamp(TODAY)
    has_today = (df["date"] == today_ts).any()

    if has_today:
        print(f"  [OK] {sector_name} 已含今天({TODAY})数据。")
        return df

    # 3. 尝试获取实时数据
    print(f"  [实时] {sector_name} 日线缺失{TODAY}，尝试实时API…")
    rt = fetch_realtime_sector(ths_code)
    if rt is None:
        print(f"  [警告] {sector_name} 实时数据获取失败，跳过今天。")
        return df

    rt_date = pd.Timestamp(rt["date"])
    if rt_date != today_ts:
        print(f"  [警告] {sector_name} 实时数据日期({rt['date']})与今天({TODAY})不一致，跳过。")
        return df

    # 4. 追加实时数据
    new_row = pd.DataFrame([rt])
    new_row["date"] = pd.Timestamp(rt["date"])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  [实时] {sector_name} 已追加今天实时数据: "
          f"开={rt['open']:.2f}, 收={rt['close']:.2f}, "
          f"高={rt['high']:.2f}, 低={rt['low']:.2f}, "
          f"涨跌幅={(rt['close']/df.iloc[-2]['close']-1)*100:+.2f}%")
    return df

if __name__ == "__main__":
    print(f"数据区间: {START} ~ {END}")

    # 第一步：拉取所有板块日线数据
    for key, (display_name, ths_name, ths_code) in SECTORS.items():
        try:
            d = fetch_sector(ths_name)
            print(f"[OK] {key}({ths_name}): {len(d)} 行, {d['date'].min().date()} ~ {d['date'].max().date()}")
        except Exception as e:
            print(f"[FAIL] {key} 日线拉取失败: {repr(e)[:150]}")
            # 尝试加载已有数据
            fpath = f"{DATA_DIR}/sector_{key}.csv"
            if os.path.exists(fpath):
                d = pd.read_csv(fpath, parse_dates=["date"])
                print(f"  → 加载已有缓存: {len(d)} 行")
            else:
                continue

        # 第二步：实时数据fallback
        d = merge_realtime(d, ths_code, key)
        d.to_csv(f"{DATA_DIR}/sector_{key}.csv", index=False, encoding="utf-8-sig")

    # 大盘指数
    for code, label in [("sh000001","上证指数"), ("sh000300","沪深300")]:
        try:
            d = fetch_index(code)
            d.to_csv(f"{DATA_DIR}/index_{label}.csv", index=False, encoding="utf-8-sig")
            print(f"[OK] {label}({code}): {len(d)} 行")
        except Exception as e:
            print(f"[FAIL] {label}: {repr(e)[:150]}")

    print("数据拉取完成。")
