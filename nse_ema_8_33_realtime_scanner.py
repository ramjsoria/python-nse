import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
import requests

# =========================
# CONFIG
# =========================
WATCHLIST: Dict[str, List[str]] = {
    "Capital Goods": ["ABB", "SIEMENS", "CGPOWER", "CUMMINSIND", "KIRLOSENG", "TIMKEN"],
    "Oil & Gas": ["BPCL", "HPCL", "IOC", "ONGC", "MRPL", "CHENNPETRO"],
    "Banks": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "LUPIN", "AUROPHARMA"],
    "FMCG": ["BRITANNIA", "DABUR", "TATACONSUM", "NESTLEIND", "CCL"],
}

ENTRY_INTERVAL = os.getenv("ENTRY_INTERVAL", "15m")
TREND_INTERVAL = os.getenv("TREND_INTERVAL", "60m")
FAST_EMA = int(os.getenv("FAST_EMA", "8"))
SLOW_EMA = int(os.getenv("SLOW_EMA", "33"))
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "180"))
PERIOD = os.getenv("PERIOD", "60d")
SHOW_ONLY_FRESH_CROSS = os.getenv("SHOW_ONLY_FRESH_CROSS", "true").lower() == "true"
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "1.2"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")
CSV_PATH = os.getenv("CSV_PATH", "ema_scanner_signals.csv")

# Telegram optional
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID", "").strip()
SEND_TELEGRAM = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# India market hours (regular session)
MARKET_START_HOUR = 9
MARKET_START_MINUTE = 15
MARKET_END_HOUR = 15
MARKET_END_MINUTE = 30


@dataclass
class Signal:
    symbol: str
    sector: str
    side: str
    last_price: float
    fast_ema: float
    slow_ema: float
    htf_fast_ema: float
    htf_slow_ema: float
    crossed_now: bool
    volume_ratio: float


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_ohlc(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        df = flatten_columns(df)
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    except Exception:
        return None


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["Close"].ewm(span=FAST_EMA, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=SLOW_EMA, adjust=False).mean()
    out["avg_vol_20"] = out["Volume"].rolling(20).mean()
    out["volume_ratio"] = out["Volume"] / out["avg_vol_20"]
    out["cross_up"] = (out["ema_fast"] > out["ema_slow"]) & (out["ema_fast"].shift(1) <= out["ema_slow"].shift(1))
    out["cross_down"] = (out["ema_fast"] < out["ema_slow"]) & (out["ema_fast"].shift(1) >= out["ema_slow"].shift(1))
    return out


def scan_symbol(symbol: str, sector: str) -> Optional[Signal]:
    entry_df = fetch_ohlc(symbol, ENTRY_INTERVAL, PERIOD)
    trend_df = fetch_ohlc(symbol, TREND_INTERVAL, PERIOD)
    if entry_df is None or trend_df is None or len(entry_df) < SLOW_EMA + 20 or len(trend_df) < SLOW_EMA + 5:
        return None

    entry_df = add_emas(entry_df)
    trend_df = add_emas(trend_df)

    e = entry_df.iloc[-1]
    t = trend_df.iloc[-1]

    if pd.isna(e["avg_vol_20"]) or e["avg_vol_20"] <= 0:
        return None

    bullish_trend = (t["Close"] > t["ema_fast"]) and (t["ema_fast"] > t["ema_slow"])
    bearish_trend = (t["Close"] < t["ema_fast"]) and (t["ema_fast"] < t["ema_slow"])

    crossed_up_now = bool(e["cross_up"])
    crossed_down_now = bool(e["cross_down"])
    volume_ratio = float(e["volume_ratio"])

    if volume_ratio < MIN_VOLUME_RATIO:
        return None

    if bullish_trend and ((crossed_up_now and SHOW_ONLY_FRESH_CROSS) or ((e["ema_fast"] > e["ema_slow"]) and not SHOW_ONLY_FRESH_CROSS)):
        return Signal(
            symbol=symbol,
            sector=sector,
            side="BUY",
            last_price=round(float(e["Close"]), 2),
            fast_ema=round(float(e["ema_fast"]), 2),
            slow_ema=round(float(e["ema_slow"]), 2),
            htf_fast_ema=round(float(t["ema_fast"]), 2),
            htf_slow_ema=round(float(t["ema_slow"]), 2),
            crossed_now=crossed_up_now,
            volume_ratio=round(volume_ratio, 2),
        )

    if bearish_trend and ((crossed_down_now and SHOW_ONLY_FRESH_CROSS) or ((e["ema_fast"] < e["ema_slow"]) and not SHOW_ONLY_FRESH_CROSS)):
        return Signal(
            symbol=symbol,
            sector=sector,
            side="SELL",
            last_price=round(float(e["Close"]), 2),
            fast_ema=round(float(e["ema_fast"]), 2),
            slow_ema=round(float(e["ema_slow"]), 2),
            htf_fast_ema=round(float(t["ema_fast"]), 2),
            htf_slow_ema=round(float(t["ema_slow"]), 2),
            crossed_now=crossed_down_now,
            volume_ratio=round(volume_ratio, 2),
        )

    return None


def run_scan() -> List[Signal]:
    results: List[Signal] = []
    for sector, symbols in WATCHLIST.items():
        for symbol in symbols:
            signal = scan_symbol(symbol, sector)
            if signal:
                results.append(signal)
    return results


def sector_strength_summary(signals: List[Signal]) -> List[Tuple[str, int, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for signal in signals:
        if signal.sector not in summary:
            summary[signal.sector] = {"BUY": 0, "SELL": 0}
        summary[signal.sector][signal.side] += 1
    ranked = []
    for sector, counts in summary.items():
        strength = counts["BUY"] - counts["SELL"]
        ranked.append((sector, counts["BUY"], strength))
    ranked.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return ranked


def print_scan(signals: List[Signal]) -> None:
    print("\n" + "=" * 100)
    print(f"EMA {FAST_EMA}/{SLOW_EMA} MTF Scanner | Entry: {ENTRY_INTERVAL} | Trend: {TREND_INTERVAL} | Vol Filter: {MIN_VOLUME_RATIO}x")
    print("=" * 100)

    if not signals:
        print("No qualifying signals found.")
        return

    ranked_sectors = sector_strength_summary(signals)
    if ranked_sectors:
        print("\nTOP SECTOR STRENGTH")
        for sector, buy_count, strength in ranked_sectors[:5]:
            print(f"- {sector:14s} | Buy Signals: {buy_count} | Strength Score: {strength}")

    buy_signals = [s for s in signals if s.side == "BUY"]
    sell_signals = [s for s in signals if s.side == "SELL"]

    if buy_signals:
        print("\nBUY SIGNALS")
        for s in buy_signals:
            freshness = "Fresh Cross" if s.crossed_now else "Trend Active"
            print(
                f"- {s.symbol:12s} | {s.sector:14s} | LTP: {s.last_price:8.2f} | "
                f"EMA8: {s.fast_ema:8.2f} | EMA33: {s.slow_ema:8.2f} | "
                f"Vol: {s.volume_ratio:4.2f}x | {freshness}"
            )

    if sell_signals:
        print("\nSELL SIGNALS")
        for s in sell_signals:
            freshness = "Fresh Cross" if s.crossed_now else "Trend Active"
            print(
                f"- {s.symbol:12s} | {s.sector:14s} | LTP: {s.last_price:8.2f} | "
                f"EMA8: {s.fast_ema:8.2f} | EMA33: {s.slow_ema:8.2f} | "
                f"Vol: {s.volume_ratio:4.2f}x | {freshness}"
            )


def export_csv(signals: List[Signal], path: str) -> None:
    file_exists = os.path.exists(path)
    timestamp = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "symbol",
                "sector",
                "side",
                "last_price",
                "fast_ema",
                "slow_ema",
                "htf_fast_ema",
                "htf_slow_ema",
                "volume_ratio",
                "crossed_now",
            ])
        for s in signals:
            writer.writerow([
                timestamp,
                s.symbol,
                s.sector,
                s.side,
                s.last_price,
                s.fast_ema,
                s.slow_ema,
                s.htf_fast_ema,
                s.htf_slow_ema,
                s.volume_ratio,
                s.crossed_now,
            ])


def build_telegram_message(signals: List[Signal]) -> str:
    now_str = datetime.now(ZoneInfo(TIMEZONE)).strftime("%d-%m-%Y %H:%M")
    if not signals:
        return f"EMA 8/33 Scanner\n{now_str}\n\nNo qualifying signals found."

    ranked_sectors = sector_strength_summary(signals)
    lines = [f"EMA 8/33 Scanner\n{now_str}"]
    if ranked_sectors:
        lines.append(f"Top sector: {ranked_sectors[0][0]}")
    lines.append("")

    buys = [s for s in signals if s.side == "BUY"][:5]
    sells = [s for s in signals if s.side == "SELL"][:5]

    if buys:
        lines.append("BUY")
        for s in buys:
            lines.append(f"{s.symbol} | {s.sector} | {s.last_price} | Vol {s.volume_ratio}x")
        lines.append("")

    if sells:
        lines.append("SELL")
        for s in sells:
            lines.append(f"{s.symbol} | {s.sector} | {s.last_price} | Vol {s.volume_ratio}x")

    return "\n".join(lines).strip()


def send_telegram_alert(message: str) -> None:
    if not SEND_TELEGRAM:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
    }
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as exc:
        print(f"Telegram send failed: {exc}")


def is_market_hours() -> bool:
    now = datetime.now(ZoneInfo(TIMEZONE))
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=MARKET_START_HOUR, minute=MARKET_START_MINUTE, second=0, microsecond=0)
    end = now.replace(hour=MARKET_END_HOUR, minute=MARKET_END_MINUTE, second=0, microsecond=0)
    return start <= now <= end


def sleep_until_market_open() -> int:
    now = datetime.now(ZoneInfo(TIMEZONE))
    next_open = now.replace(hour=MARKET_START_HOUR, minute=MARKET_START_MINUTE, second=0, microsecond=0)

    if now.weekday() >= 5:
        days_ahead = 7 - now.weekday()
        next_open = next_open + timedelta(days=days_ahead)
    elif now.time() > next_open.time() and now.replace(hour=MARKET_END_HOUR, minute=MARKET_END_MINUTE).time() < now.time():
        next_open = next_open + timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open = next_open + timedelta(days=1)
    elif now.time() < next_open.time():
        pass
    else:
        next_open = now

    wait_seconds = max(30, int((next_open - now).total_seconds()))
    return wait_seconds


def main() -> None:
    print("Starting NSE EMA 8/33 realtime scanner...")
    print(f"Entry TF: {ENTRY_INTERVAL} | Trend TF: {TREND_INTERVAL} | Refresh: {REFRESH_SECONDS}s | Vol Filter: {MIN_VOLUME_RATIO}x")
    print(f"Market hours only: 09:15 to 15:30 {TIMEZONE}")
    print(f"CSV export: {CSV_PATH}")
    print(f"Telegram alerts: {'ON' if SEND_TELEGRAM else 'OFF'}")

    last_sent_fingerprints = set()

    while True:
        try:
            if not is_market_hours():
                wait_seconds = sleep_until_market_open()
                print(f"Outside market hours. Sleeping for {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                continue

            signals = run_scan()
            print_scan(signals)
            export_csv(signals, CSV_PATH)

            current_fingerprints = {
                (s.symbol, s.side, s.crossed_now, s.last_price) for s in signals
            }
            new_signals = current_fingerprints - last_sent_fingerprints
            if new_signals and SEND_TELEGRAM:
                filtered = [
                    s for s in signals if (s.symbol, s.side, s.crossed_now, s.last_price) in new_signals
                ]
                send_telegram_alert(build_telegram_message(filtered))
            last_sent_fingerprints = current_fingerprints

        except KeyboardInterrupt:
            print("\nScanner stopped.")
            break
        except Exception as exc:
            print(f"Error during scan: {exc}")

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
