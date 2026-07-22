#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股板块评估系统 · 数据拉取模块
拉取 6 个板块(同花顺行业指数) + 大盘基准(新浪指数) 2021-01-01 ~ 今天的日线数据。
"""
import akshare as ak, pandas as pd, time, os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
END = pd.Timestamp.today().strftime("%Y%m%d")
START = "20210101"

SECTORS = {
    "半导体": "半导体", "消费电子": "消费电子",
    "通信": "通信设备", "电池": "电池",
    "电力": "电力", "煤炭": "煤炭开采加工",
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

def fetch_sector(name):
    df = retry(lambda: ak.stock_board_industry_index_ths(
        symbol=name, start_date=START, end_date=END))
    df = df.rename(columns={"日期":"date","开盘价":"open","最高价":"high",
                            "最低价":"low","收盘价":"close","成交量":"volume","成交额":"amount"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    return df[df["date"] >= "2021-01-01"]

def fetch_index(code):
    df = retry(lambda: ak.stock_zh_index_daily(symbol=code))
    df = df.rename(columns={"date":"date","open":"open","high":"high",
                            "low":"low","close":"close","volume":"volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df["amount"] = float("nan")
    return df[df["date"] >= "2021-01-01"]

if __name__ == "__main__":
    print(f"数据区间: {START} ~ {END}")
    for key, name in SECTORS.items():
        try:
            d = fetch_sector(name)
            d.to_csv(f"{DATA_DIR}/sector_{key}.csv", index=False, encoding="utf-8-sig")
            print(f"[OK] {key}: {len(d)} 行, {d['date'].min().date()} ~ {d['date'].max().date()}")
        except Exception as e:
            print(f"[FAIL] {key}: {repr(e)[:150]}")
    for code, label in [("sh000001","上证指数"), ("sh000300","沪深300")]:
        try:
            d = fetch_index(code)
            d.to_csv(f"{DATA_DIR}/index_{label}.csv", index=False, encoding="utf-8-sig")
            print(f"[OK] {label}({code}): {len(d)} 行")
        except Exception as e:
            print(f"[FAIL] {label}: {repr(e)[:150]}")
    print("数据拉取完成。")
