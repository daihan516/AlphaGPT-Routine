#!/usr/bin/env python3
"""抓取深交所两融数据（按日），缓存成单个 parquet 供 v2 使用。

背景：仓库自带的 margin_balance/ 只有沪市（stock_margin_detail_sse）。
深市股票在 v2 的 MARGIN_NET 因子上会恒为 0，导致搜索偏向"只赚沪市"的公式。

用法：
    .venv/bin/python fetch_margin_szse.py              # 增量抓取（默认 2022 至今）
    START_DATE=20200101 .venv/bin/python fetch_margin_szse.py
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

CACHE = "margin_szse_cache.parquet"
START_DATE = os.environ.get("START_DATE", "20220101")
WORKERS = int(os.environ.get("WORKERS", "6"))


def trading_dates():
    """交易日 = 沪市两融缓存里存在的日期（沪深交易日一致）"""
    fs = os.listdir("margin_balance")
    ds = sorted(f[:8] for f in fs
                if f.endswith("_margin_data.parquet") and f[:8] >= START_DATE)
    return ds


def fetch_one(date):
    import akshare as ak
    for attempt in range(3):
        try:
            df = ak.stock_margin_detail_szse(date=date)
            if df is None or df.empty:
                return date, None
            out = df[["证券代码", "融资余额", "融资买入额"]].copy()
            out.columns = ["code", "balance", "buy"]
            out["date"] = date
            return date, out
        except Exception:
            time.sleep(1.0 + attempt)
    return date, "FAIL"


def main():
    dates = trading_dates()
    done = {}
    if os.path.exists(CACHE):
        old = pd.read_parquet(CACHE)
        done = {d: None for d in old["date"].unique()}
        print(f"已有缓存 {len(done)} 天")
    todo = [d for d in dates if d not in done]
    print(f"需抓取 {len(todo)} 天（共 {len(dates)} 天），并发 {WORKERS}")

    frames, fails = [], []

    def flush():
        """增量落盘，避免中途崩溃白跑"""
        if not frames:
            return
        new = pd.concat(frames, ignore_index=True)
        if os.path.exists(CACHE):
            new = pd.concat([pd.read_parquet(CACHE), new], ignore_index=True)
        new = new.drop_duplicates(subset=["date", "code"], keep="last")
        new = new.sort_values(["date", "code"])
        new.to_parquet(CACHE)
        frames.clear()
        return new

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, d): d for d in todo}
        for n, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc="抓取深市两融"), 1):
            d, res = fut.result()
            if res is None or isinstance(res, str):     # 非交易日 / 抓取失败
                fails.append(d)
            else:
                frames.append(res)
            if n % 100 == 0:
                flush()

    saved = flush()
    if saved is not None:
        print(f"已写入 {CACHE}: {saved.shape}, 覆盖 {saved['date'].nunique()} 天")
    if fails:
        print(f"失败 {len(fails)} 天（非交易日或限流）: {fails[:5]} ...")


if __name__ == "__main__":
    main()
