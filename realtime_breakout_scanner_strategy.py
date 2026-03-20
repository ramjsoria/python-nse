import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ============================================================
# REALTIME BREAKOUT SCANNER FOR NSE STOCKS
# Strategy logic:
# - Stock above EMA 20
# - Stock above EMA 50
# - EMA 20 above EMA 50
# - RSI strong
# - Volume rising before breakout
# - Tight box / contraction range
# - Sector leaders also bullish
# - Market is not weak
# - Optional Telegram alerts + CSV export
# ============================================================

# ---------------------- CONFIG ----------------------
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")
ENTRY_INTERVAL = os.getenv("ENTRY_INTERVAL", "15m")
TREND_INTERVAL = os.getenv("TREND_INTERVAL", "60m")
MARKET_INTERVAL = os.getenv("MARKET_INTERVAL", "15m")
PERIOD = os.getenv("PERIOD", "60d")
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "180"))
CSV_PATH = os.getenv("CSV_PATH", "breakout_scanner_signals.csv")
MIN_RSI = float(os.getenv("MIN_RSI", "58"))
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "1.3"))
BOX_LOOKBACK = int(os.getenv("BOX_LOOKBACK", "15"))
MAX_BOX_RANGE_PCT = float(os.getenv("MAX_BOX_RANGE_PCT", "0.10"))
NEAR_BREAKOUT_PCT = float(os.getenv("NEAR_BREAKOUT_PCT", "0.97"))
SHOW_ONLY_NEW = os.getenv("SHOW_ONLY_NEW", "true").lower() == "true"

# Telegram optional
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
SEND_TELEGRAM = bool(BOT_TOKEN and CHAT_ID)

# India market hours
MARKET_START_HOUR = 9
MARKET_START_MINUTE = 15
MARKET_END_HOUR = 15
MARKET_END_MINUTE = 30

# Broad market filter symbols
MARKET_SYMBOLS = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

# Sector leaders define whether sector is bullish.
SECTOR_LEADERS: Dict[str, List[str]] = {
    "Capital Goods": ["ABB", "SIEMENS", "CGPOWER", "CUMMINSIND"],
    "Oil & Gas": ["BPCL", "HPCL", "IOC", "ONGC"],
    "Banks": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK"],
    "Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "LUPIN"],
    "FMCG": ["BRITANNIA", "NESTLEIND", "DABUR", "TATACONSUM"],
    "Auto Ancillary": ["BOSCHLTD", "MOTHERSON", "BHARATFORG", "TIINDIA"],
}

# Watchlist to scan.
WATCHLIST: Dict[str, List[str]] = {
    "Capital Goods": ["ABB", "SIEMENS", "CGPOWER", "CUMMINSIND", "KIRLOSENG", "TIMKEN"],
    "Oil & Gas": ["BPCL", "HPCL", "IOC", "ONGC", "MRPL", "CHENNPETRO"],
    "Banks": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    "Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "LUPIN", "AUROPHARMA"],
    "FMCG": ["BRITANNIA", "DABUR", "TATACONSUM", "NESTLEIND", "CCL"],
    "Auto Ancillary": ["WHEELS", "BOSCHLTD", "MOTHERSON", "BHARATFORG", "TIINDIA"],
}


@dataclass
class ScanSignal:
    symbol: str
    sector: str
    signal_type: str
    last_price: float
    entry_price: float
    stop_loss: float
    breakout_level: float
    support_level: float
    ema20: float
    ema50: float
    rsi: float
    volume_ratio: float
    box_range_pct: float
    sector_bullish: bool
    market_ok: bool
    strength_score: float


# ---------------------- DATA HELPERS ----------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_ohlc(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        df = flatten_columns(df)
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    except Exception:
        return None


def fetch_nse_stock(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    return fetch_ohlc(f"{symbol}.NS", interval, period)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ema(out["Close"], 20)
    out["ema50"] = ema(out["Close"], 50)
    out["rsi14"] = rsi(out["Close"], 14)
    out["avg_vol20"] = out["Volume"].rolling(20).mean()
    out["volume_ratio"] = out["Volume"] / out["avg_vol20"]
    out["box_high"] = out["High"].rolling(BOX_LOOKBACK).max()
    out["box_low"] = out["Low"].rolling(BOX_LOOKBACK).min()
    out["box_range_pct"] = (out["box_high"] / out["box_low"]) - 1
    out["cross_above_box"] = (out["Close"] > out["box_high"].shift(1))
    return out


# ---------------------- STRATEGY FILTERS ----------------------
def is_market_trending_ok() -> bool:
    bullish_count = 0
    checked = 0
    for _, ticker in MARKET_SYMBOLS.items():
        df = fetch_ohlc(ticker, MARKET_INTERVAL, PERIOD)
        if df is None or len(df) < 60:
            continue
        df = add_indicators(df)
        last = df.iloc[-1]
        checked += 1
        if (
            last["Close"] > last["ema20"]
            and last["Close"] > last["ema50"]
            and last["ema20"] > last["ema50"]
        ):
            bullish_count += 1
    return checked > 0 and bullish_count >= max(1, checked // 2)


def sector_is_bullish(sector: str) -> bool:
    leaders = SECTOR_LEADERS.get(sector, [])
    if not leaders:
        return True
    bullish = 0
    total = 0
    for symbol in leaders:
        df = fetch_nse_stock(symbol, TREND_INTERVAL, PERIOD)
        if df is None or len(df) < 60:
            continue
        df = add_indicators(df)
        last = df.iloc[-1]
        total += 1
        if (
            last["Close"] > last["ema20"]
            and last["Close"] > last["ema50"]
            and last["ema20"] > last["ema50"]
            and last["rsi14"] >= 55
        ):
            bullish += 1
    return total > 0 and bullish >= max(1, (total + 1) // 2)


def calculate_strength_score(last_row: pd.Series, sector_ok: bool, market_ok: bool) -> float:
    score = 0.0
    score += 25 if last_row["Close"] > last_row["ema20"] else 0
    score += 25 if last_row["Close"] > last_row["ema50"] and last_row["ema20"] > last_row["ema50"] else 0
    score += min(float(last_row["rsi14"]) / 70.0, 1.2) * 15
    score += min(float(last_row["volume_ratio"]), 3.0) * 10
    tightness = 1 - min(float(last_row["box_range_pct"]) / MAX_BOX_RANGE_PCT, 1.0)
    score += tightness * 10
    score += 10 if float(last_row["Close"]) >= NEAR_BREAKOUT_PCT * float(last_row["box_high"]) else 0
    score += 3 if sector_ok else 0
    score += 2 if market_ok else 0
    return round(score, 2)


def scan_symbol(symbol: str, sector: str, market_ok: bool, sector_ok: bool) -> Optional[ScanSignal]:
    df = fetch_nse_stock(symbol, ENTRY_INTERVAL, PERIOD)
    if df is None or len(df) < max(60, BOX_LOOKBACK + 20):
        return None

    df = add_indicators(df)
    last = df.iloc[-1]

    needed = [
        last["ema20"], last["ema50"], last["rsi14"], last["avg_vol20"],
        last["volume_ratio"], last["box_high"], last["box_low"], last["box_range_pct"],
    ]
    if any(pd.isna(v) for v in needed):
        return None

    strong_trend = (
        last["Close"] > last["ema20"]
        and last["Close"] > last["ema50"]
        and last["ema20"] > last["ema50"]
    )
    strong_rsi = last["rsi14"] >= MIN_RSI
    strong_volume = last["volume_ratio"] >= MIN_VOLUME_RATIO
    tight_box = last["box_range_pct"] <= MAX_BOX_RANGE_PCT
    near_breakout = last["Close"] >= NEAR_BREAKOUT_PCT * last["box_high"]

    if not (strong_trend and strong_rsi and strong_volume and tight_box and near_breakout and sector_ok and market_ok):
        return None

    breakout_level = round(float(last["box_high"]), 2)
    support_level = round(float(last["box_low"]), 2)
    last_price = round(float(last["Close"]), 2)
    entry_price = round(max(last_price, breakout_level), 2)
    stop_loss = round(min(float(last["ema20"]), support_level) * 0.995, 2)

    if last_price > breakout_level:
        signal_type = "BREAKOUT_ACTIVE"
    else:
        signal_type = "BREAKOUT_READY"

    return ScanSignal(
        symbol=symbol,
        sector=sector,
        signal_type=signal_type,
        last_price=last_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        breakout_level=breakout_level,
        support_level=support_level,
        ema20=round(float(last["ema20"]), 2),
        ema50=round(float(last["ema50"]), 2),
        rsi=round(float(last["rsi14"]), 2),
        volume_ratio=round(float(last["volume_ratio"]), 2),
        box_range_pct=round(float(last["box_range_pct"]) * 100, 2),
        sector_bullish=sector_ok,
        market_ok=market_ok,
        strength_score=calculate_strength_score(last, sector_ok, market_ok),
    )


# ---------------------- OUTPUTS ----------------------
def run_scan() -> List[ScanSignal]:
    market_ok = is_market_trending_ok()
    sector_cache: Dict[str, bool] = {sector: sector_is_bullish(sector) for sector in WATCHLIST.keys()}

    results: List[ScanSignal] = []
    for sector, symbols in WATCHLIST.items():
        sector_ok = sector_cache.get(sector, True)
        for symbol in symbols:
            try:
                signal = scan_symbol(symbol, sector, market_ok, sector_ok)
                if signal:
                    results.append(signal)
            except Exception as exc:
                print(f"Failed on {symbol}: {exc}")

    results.sort(key=lambda x: x.strength_score, reverse=True)
    return results


def sector_ranking(signals: List[ScanSignal]) -> List[Tuple[str, int, float]]:
    data: Dict[str, List[float]] = {}
    for signal in signals:
        data.setdefault(signal.sector, []).append(signal.strength_score)
    ranked = [(sector, len(scores), round(sum(scores), 2)) for sector, scores in data.items()]
    ranked.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return ranked


def export_csv(signals: List[ScanSignal], path: str) -> None:
    file_exists = os.path.exists(path)
    now_str = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "sector", "signal_type", "last_price", "entry_price",
                "stop_loss", "breakout_level", "support_level", "ema20", "ema50", "rsi",
                "volume_ratio", "box_range_pct", "sector_bullish", "market_ok", "strength_score"
            ])
        for s in signals:
            writer.writerow([
                now_str, s.symbol, s.sector, s.signal_type, s.last_price, s.entry_price,
                s.stop_loss, s.breakout_level, s.support_level, s.ema20, s.ema50, s.rsi,
                s.volume_ratio, s.box_range_pct, s.sector_bullish, s.market_ok, s.strength_score
            ])


def build_telegram_message(signals: List[ScanSignal]) -> str:
    now_str = datetime.now(ZoneInfo(TIMEZONE)).strftime("%d-%m-%Y %H:%M")
    if not signals:
        return f"Breakout Scanner\n{now_str}\n\nNo qualifying stocks right now."

    ranked = sector_ranking(signals)
    lines = [f"Breakout Scanner\n{now_str}"]
    if ranked:
        lines.append(f"Top sector: {ranked[0][0]}")
    lines.append("")
    for s in signals[:5]:
        lines.append(
            f"{s.symbol} | {s.sector}\n"
            f"Type: {s.signal_type}\n"
            f"LTP: {s.last_price} | Entry: {s.entry_price}\n"
            f"SL: {s.stop_loss} | RSI: {s.rsi} | Vol: {s.volume_ratio}x\n"
            f"Score: {s.strength_score}"
        )
        lines.append("")
    return "\n".join(lines).strip()


def send_telegram_alert(message: str) -> None:
    if not SEND_TELEGRAM:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as exc:
        print(f"Telegram send failed: {exc}")


def print_signals(signals: List[ScanSignal]) -> None:
    print("\n" + "=" * 120)
    print("REALTIME BREAKOUT SCANNER | EMA 20 / EMA 50 | RSI | VOLUME | BOX RANGE | SECTOR FILTER | MARKET FILTER")
    print("=" * 120)
    if not signals:
        print("No qualifying stocks right now.")
        return

    ranked = sector_ranking(signals)
    print("\nSECTOR RANKING")
    for sector, count, score in ranked[:5]:
        print(f"- {sector:15s} | Signals: {count:2d} | Sector Score: {score:6.2f}")

    print("\nTOP SIGNALS")
    for s in signals[:10]:
        print(
            f"- {s.symbol:12s} | {s.sector:15s} | {s.signal_type:15s} | "
            f"LTP: {s.last_price:8.2f} | Entry: {s.entry_price:8.2f} | SL: {s.stop_loss:8.2f} | "
            f"EMA20: {s.ema20:8.2f} | EMA50: {s.ema50:8.2f} | RSI: {s.rsi:6.2f} | "
            f"Vol: {s.volume_ratio:4.2f}x | Box: {s.box_range_pct:5.2f}% | Score: {s.strength_score:6.2f}"
        )


# ---------------------- MARKET HOURS ----------------------
def is_market_hours() -> bool:
    now = datetime.now(ZoneInfo(TIMEZONE))
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=MARKET_START_HOUR, minute=MARKET_START_MINUTE, second=0, microsecond=0)
    end = now.replace(hour=MARKET_END_HOUR, minute=MARKET_END_MINUTE, second=0, microsecond=0)
    return start <= now <= end


def seconds_until_market_open() -> int:
    now = datetime.now(ZoneInfo(TIMEZONE))
    next_open = now.replace(hour=MARKET_START_HOUR, minute=MARKET_START_MINUTE, second=0, microsecond=0)
    if now.weekday() >= 5:
        days_ahead = 7 - now.weekday()
        next_open = next_open + timedelta(days=days_ahead)
    elif now.time() > now.replace(hour=MARKET_END_HOUR, minute=MARKET_END_MINUTE).time():
        next_open = next_open + timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
    elif now.time() < next_open.time():
        pass
    else:
        next_open = now
    return max(30, int((next_open - now).total_seconds()))


# ---------------------- MAIN LOOP ----------------------
def main() -> None:
    print("Starting realtime breakout scanner...")
    print(f"Entry interval: {ENTRY_INTERVAL} | Trend interval: {TREND_INTERVAL} | Refresh: {REFRESH_SECONDS}s")
    print(f"Filters: RSI >= {MIN_RSI}, Volume >= {MIN_VOLUME_RATIO}x, Box <= {MAX_BOX_RANGE_PCT * 100:.1f}%")
    print(f"Telegram alerts: {'ON' if SEND_TELEGRAM else 'OFF'} | CSV: {CSV_PATH}")

    sent_fingerprints = set()

    while True:
        try:
            if not is_market_hours():
                wait_seconds = seconds_until_market_open()
                print(f"Outside market hours. Sleeping for {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                continue

            signals = run_scan()
            print_signals(signals)
            export_csv(signals, CSV_PATH)

            current = {(s.symbol, s.signal_type, s.last_price) for s in signals}
            if SHOW_ONLY_NEW:
                new_items = current - sent_fingerprints
                new_signals = [s for s in signals if (s.symbol, s.signal_type, s.last_price) in new_items]
            else:
                new_signals = signals

            if new_signals and SEND_TELEGRAM:
                send_telegram_alert(build_telegram_message(new_signals))

            sent_fingerprints = current
            time.sleep(REFRESH_SECONDS)

        except KeyboardInterrupt:
            print("\nScanner stopped.")
            break
        except Exception as exc:
            print(f"Error during scan: {exc}")
            time.sleep(30)


if __name__ == "__main__":
    main()
