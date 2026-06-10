#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_agent.py — v5
Personal market signal agent (XTB manual execution, Yahoo Finance data).

What it does each run:
  1. Determines the session wave (EU open 09:00 / EU conf 10:00 / US open 15:30 /
     US conf 16:30 CET) — or handles off-schedule runs gracefully.
  2. Pulls daily + intraday data for the 14-ticker watchlist via yfinance.
  3. Computes exact RSI-14, MA20/50/200, ATR-14 and assigns a verdict
     (DIP / MOMENTUM / NEUTRAL / AVOID) with 1.5x ATR stop and 2.5x ATR target.
  4. Two-wave logic: open-wave signals are provisional; the confirmation wave
     an hour later marks signals CONFIRMED if still valid (filters opening noise).
  5. Daily Opportunities scanner: ranks a universe of liquid, stable large-caps
     you do NOT hold (IBM / INTC style names) by intraday movement + setup
     quality, surfaces the top 5 with entry/stop/target.
  6. CFD picks: scores ~27 liquid EU+US names, surfaces top 3.
  7. Writes everything to a self-contained dashboard.html and persists state
     to session_state.json.

Journal (trades.json):
  python market_agent.py journal add  --ticker NVDA --type dip --entry 142.5 --size 10
  python market_agent.py journal close --id 3 --exit 151.2
  python market_agent.py journal stats

Scan:
  python market_agent.py                # auto-detect wave from current CET time
  python market_agent.py --wave us_open # force a wave
  python market_agent.py --demo         # synthetic data, no network (testing)

This is a decision aid, not gospel. Verify everything on XTB before acting.
"""

import argparse
import json
import math
import os
import random
import sys
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "session_state.json"
TRADES_FILE = BASE_DIR / "trades.json"
DASHBOARD_FILE = BASE_DIR / "dashboard.html"

# Held watchlist — these are scanned for signals and EXCLUDED from opportunities.
WATCHLIST_EU = ["CSG.AS", "IWDA.L", "CSPX.L"]
WATCHLIST_US = ["GOOGL", "PLTR", "NVDA", "NOW", "SNDK", "MU",
                "CRWV", "INOD", "SOFI", "PGY", "META"]
WATCHLIST = WATCHLIST_EU + WATCHLIST_US

# Daily Opportunities universe: liquid, stable, non-held large caps.
# IBM / INTC are the reference profile: big, boring, but they move intraday.
OPPORTUNITY_UNIVERSE = [
    "IBM", "INTC", "AMD", "CSCO", "ORCL", "QCOM", "TXN", "MSFT", "AAPL",
    "JPM", "BAC", "MS", "GS", "V", "MA",
    "XOM", "CVX", "KO", "PEP", "PG", "WMT", "COST",
    "DIS", "NKE", "MCD", "T", "VZ", "PFE", "MRK", "JNJ",
    "GE", "CAT", "BA", "F", "GM", "UPS",
]

# CFD scanner universe (~27 liquid EU + US names).
CFD_UNIVERSE = [
    # US megacaps / movers
    "AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "GOOGL", "AMD", "NFLX",
    "INTC", "BA", "JPM", "XOM", "DIS", "PLTR", "COIN", "UBER",
    # EU liquid names
    "ASML.AS", "SAP.DE", "SIE.DE", "AIR.PA", "MC.PA", "OR.PA",
    "NESN.SW", "NOVO-B.CO", "SHEL.L", "HSBA.L",
]

RSI_PERIOD = 14
ATR_PERIOD = 14
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.5

# CET/CEST schedule (the Task Scheduler .bat files fire at these times)
WAVES = {
    "eu_open":  {"label": "EU Open Wave (09:00 CET)",      "markets": ["EU"]},
    "eu_conf":  {"label": "EU Confirmation (10:00 CET)",   "markets": ["EU"]},
    "us_open":  {"label": "US Open Wave (15:30 CET)",      "markets": ["US"]},
    "us_conf":  {"label": "US Confirmation (16:30 CET)",   "markets": ["US"]},
}


def now_cet():
    """Current time in CET/CEST without external tz dependencies."""
    utc = datetime.now(timezone.utc)
    # CEST (UTC+2) roughly Apr–Oct, CET (UTC+1) otherwise — good enough for
    # session gating; the .bat schedule is the real trigger.
    month = utc.month
    offset = 2 if 4 <= month <= 10 else 1
    return utc + timedelta(hours=offset)


def detect_wave(t=None):
    """Map current CET time to the nearest wave; None if off-schedule."""
    t = t or now_cet()
    hm = t.hour * 60 + t.minute
    table = [("eu_open", 9 * 60), ("eu_conf", 10 * 60),
             ("us_open", 15 * 60 + 30), ("us_conf", 16 * 60 + 30)]
    for name, mins in table:
        if abs(hm - mins) <= 25:          # within 25 min of a scheduled wave
            return name
    return None


def market_open_now(market, t=None):
    """Rough open/closed check in CET (weekdays only)."""
    t = t or now_cet()
    if t.weekday() >= 5:
        return False
    hm = t.hour * 60 + t.minute
    if market == "EU":
        return 9 * 60 <= hm <= 17 * 60 + 30
    return 15 * 60 + 30 <= hm <= 22 * 60   # US ~15:30–22:00 CET


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------

def rsi(closes, period=RSI_PERIOD):
    """Wilder's RSI on a list of closes. Returns latest value or None."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def atr(highs, lows, closes, period=ATR_PERIOD):
    """Wilder's ATR. Returns latest value or None."""
    n = len(closes)
    if n < period + 1:
        return None
    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    a = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        a = (a * (period - 1) + trs[i]) / period
    return a


# ----------------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------------

def fetch_history(tickers, demo=False):
    """
    Returns {ticker: {"closes": [...], "highs": [...], "lows": [...],
                      "price": float, "prev_close": float, "volume": float,
                      "avg_volume": float, "currency": str}}
    Missing/failed tickers are simply absent from the dict.
    """
    if demo:
        return _demo_history(tickers)

    import yfinance as yf
    out = {}
    data = yf.download(tickers, period="12mo", interval="1d",
                       group_by="ticker", auto_adjust=False,
                       progress=False, threads=True)
    for t in tickers:
        try:
            df = data[t] if len(tickers) > 1 else data
            df = df.dropna(subset=["Close"])
            if len(df) < 60:
                continue
            closes = df["Close"].tolist()
            highs = df["High"].tolist()
            lows = df["Low"].tolist()
            vols = df["Volume"].tolist()
            out[t] = {
                "closes": closes, "highs": highs, "lows": lows,
                "price": float(closes[-1]),
                "prev_close": float(closes[-2]),
                "volume": float(vols[-1]) if vols else 0.0,
                "avg_volume": (sum(vols[-20:]) / min(20, len(vols))) if vols else 0.0,
            }
        except Exception as e:
            print(f"  ! {t}: {e}")
    # Try to refine the last price with a recent intraday quote
    try:
        intraday = yf.download(tickers, period="1d", interval="5m",
                               group_by="ticker", progress=False, threads=True)
        for t in list(out.keys()):
            try:
                df = intraday[t] if len(tickers) > 1 else intraday
                df = df.dropna(subset=["Close"])
                if len(df):
                    out[t]["price"] = float(df["Close"].iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    return out


def _demo_history(tickers):
    """Synthetic but realistic-looking data for offline testing (--demo)."""
    rng = random.Random(42)
    out = {}
    for t in tickers:
        base = rng.uniform(20, 600)
        closes, highs, lows = [], [], []
        price = base
        trend = rng.uniform(-0.001, 0.002)
        for _ in range(260):
            price *= 1 + trend + rng.gauss(0, 0.018)
            hi = price * (1 + abs(rng.gauss(0, 0.008)))
            lo = price * (1 - abs(rng.gauss(0, 0.008)))
            closes.append(price)
            highs.append(hi)
            lows.append(lo)
        out[t] = {
            "closes": closes, "highs": highs, "lows": lows,
            "price": closes[-1] * (1 + rng.gauss(0, 0.015)),
            "prev_close": closes[-1],
            "volume": rng.uniform(1e6, 6e7),
            "avg_volume": rng.uniform(1e6, 6e7),
        }
    return out


# ----------------------------------------------------------------------------
# Signal engine
# ----------------------------------------------------------------------------

def analyze(t, h, allow_short=False):
    """Compute indicators + verdict for one ticker. allow_short enables
    short-side setups (used for opportunities + CFD, not held watchlist)."""
    closes, highs, lows = h["closes"], h["highs"], h["lows"]
    price = h["price"]
    r = rsi(closes)
    ma20, ma50, ma200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    a = atr(highs, lows, closes)
    day_chg = (price / h["prev_close"] - 1) * 100 if h["prev_close"] else 0.0

    verdict, reason = "NEUTRAL", "No clear setup."
    uptrend = ma200 is not None and price > ma200
    pullback_3d = (closes[-4] - price) if len(closes) >= 4 else 0.0

    if (allow_short and ma200 is not None and price < ma200
            and r is not None and r > 62):
        verdict, reason = "SHORT", (f"Bear rally fade: RSI {r:.0f} overbought "
                                    "below MA200 — bounce into a broken trend.")
    elif (allow_short and ma20 and ma50 and price < ma20 < ma50
          and ma200 is not None and price < ma200
          and r is not None and 30 <= r <= 50 and day_chg < 0):
        verdict, reason = "SHORT", (f"Downtrend momentum: price < MA20 < MA50 < trend, "
                                    f"RSI {r:.0f}, red on the day ({day_chg:.1f}%).")
    elif ma200 is not None and price < ma200 and (ma50 is None or price < ma50):
        verdict, reason = "AVOID", "Below MA200 and MA50 — broken trend, falling knife risk."
    elif r is not None and r < 32:
        verdict, reason = "DIP", f"RSI {r:.0f} oversold."
        if uptrend:
            reason += " Long-term uptrend intact (above MA200)."
    elif uptrend and r is not None and r < 45 and a and pullback_3d > a:
        verdict, reason = "DIP", (f"Sharp pullback in uptrend: ~{pullback_3d:.1f} off "
                                  f"in 3 sessions (> 1 ATR), RSI {r:.0f}.")
    elif (ma20 and ma50 and price > ma20 > ma50
          and r is not None and 50 <= r <= 70 and day_chg > 0):
        verdict, reason = "MOMENTUM", (f"Price > MA20 > MA50, RSI {r:.0f}, "
                                       f"green on the day (+{day_chg:.1f}%).")

    if verdict == "SHORT":
        sl = price + SL_ATR_MULT * a if a else None   # stop ABOVE entry
        tp = price - TP_ATR_MULT * a if a else None   # target BELOW entry
    else:
        sl = price - SL_ATR_MULT * a if a else None
        tp = price + TP_ATR_MULT * a if a else None
    return {
        "ticker": t, "price": round(price, 2), "day_chg": round(day_chg, 2),
        "rsi": round(r, 1) if r is not None else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma50": round(ma50, 2) if ma50 else None,
        "ma200": round(ma200, 2) if ma200 else None,
        "atr": round(a, 2) if a else None,
        "sl": round(sl, 2) if sl else None,
        "tp": round(tp, 2) if tp else None,
        "verdict": verdict, "reason": reason,
    }


def scan_opportunities(history, held):
    """
    Daily Opportunities: liquid non-held large caps with notable intraday
    movement and a workable setup. Score = movement + liquidity + setup quality.
    """
    out = []
    for t, h in history.items():
        if t in held:
            continue
        sig = analyze(t, h, allow_short=True)
        move = abs(sig["day_chg"])
        if move < 1.0:                      # needs to actually be moving today
            continue
        liquidity = 1.0
        if h.get("avg_volume"):
            liquidity = min(2.0, max(0.5, h["volume"] / h["avg_volume"]))
        setup_bonus = {"DIP": 2.0, "MOMENTUM": 1.5, "SHORT": 2.0,
                       "NEUTRAL": 0.0, "AVOID": -3.0}[sig["verdict"]]
        score = move * 1.5 + liquidity + setup_bonus
        direction = "LONG"
        if sig["verdict"] == "SHORT":
            direction = "SHORT"
        elif sig["verdict"] == "AVOID" or (sig["day_chg"] < 0 and sig["verdict"] == "NEUTRAL"):
            # red day with no qualifying setup either way → watch only
            direction = "WATCH"
        sig.update({"score": round(score, 2), "direction": direction,
                    "vol_ratio": round(liquidity, 2)})
        out.append(sig)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:5]


def scan_cfd(history):
    """CFD picks: same engine, scored for short-horizon tradability, top 3."""
    out = []
    for t, h in history.items():
        sig = analyze(t, h, allow_short=True)
        move = abs(sig["day_chg"])
        atr_pct = (sig["atr"] / sig["price"] * 100) if sig["atr"] and sig["price"] else 0
        setup = {"DIP": 2.5, "SHORT": 2.5, "MOMENTUM": 2.0,
                 "NEUTRAL": 0.5, "AVOID": 0.0}[sig["verdict"]]
        score = move + atr_pct * 0.8 + setup
        sig.update({"score": round(score, 2)})
        out.append(sig)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:3]


# ----------------------------------------------------------------------------
# Two-wave state
# ----------------------------------------------------------------------------

def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def apply_two_wave(signals, wave, state):
    """
    Open waves store provisional signals. Confirmation waves compare against
    the morning's open wave: a DIP/MOMENTUM still standing an hour later is
    CONFIRMED; one that vanished is flagged as opening noise.
    """
    today = now_cet().strftime("%Y-%m-%d")
    if wave in ("eu_open", "us_open"):
        key = "eu" if wave == "eu_open" else "us"
        state.setdefault(today, {})[f"{key}_open"] = {
            s["ticker"]: s["verdict"] for s in signals
        }
        for s in signals:
            if s["verdict"] in ("DIP", "MOMENTUM"):
                s["status"] = "PROVISIONAL"
                s["reason"] += " (Open wave — wait for confirmation.)"
            else:
                s["status"] = "—"
    elif wave in ("eu_conf", "us_conf"):
        key = "eu" if wave == "eu_conf" else "us"
        prior = state.get(today, {}).get(f"{key}_open", {})
        for s in signals:
            earlier = prior.get(s["ticker"])
            if s["verdict"] in ("DIP", "MOMENTUM"):
                if earlier == s["verdict"]:
                    s["status"] = "CONFIRMED"
                    s["reason"] += " Held since the open wave — confirmed."
                else:
                    s["status"] = "NEW (1h)"
            elif earlier in ("DIP", "MOMENTUM"):
                s["status"] = "FADED"
                s["reason"] += f" Was {earlier} at the open — opening noise, signal faded."
            else:
                s["status"] = "—"
    else:
        for s in signals:
            s["status"] = "OFF-SCHEDULE"
    return signals


# ----------------------------------------------------------------------------
# Trade journal
# ----------------------------------------------------------------------------

def journal_add(args):
    trades = load_json(TRADES_FILE, [])
    trade = {
        "id": (max((t["id"] for t in trades), default=0) + 1),
        "ticker": args.ticker.upper(), "type": args.type,
        "entry": args.entry, "size": args.size,
        "opened": now_cet().strftime("%Y-%m-%d %H:%M"),
        "exit": None, "closed": None, "pnl": None,
    }
    trades.append(trade)
    save_json(TRADES_FILE, trades)
    print(f"Logged trade #{trade['id']}: {trade['ticker']} {trade['type']} @ {trade['entry']}")


def journal_close(args):
    trades = load_json(TRADES_FILE, [])
    for t in trades:
        if t["id"] == args.id and t["exit"] is None:
            t["exit"] = args.exit
            t["closed"] = now_cet().strftime("%Y-%m-%d %H:%M")
            t["pnl"] = round((args.exit - t["entry"]) * t["size"], 2)
            save_json(TRADES_FILE, trades)
            print(f"Closed #{t['id']} {t['ticker']}: PnL {t['pnl']:+}")
            return
    print(f"No open trade with id {args.id}.")


def journal_stats(_args=None, quiet=False):
    trades = load_json(TRADES_FILE, [])
    closed = [t for t in trades if t["pnl"] is not None]
    stats = {}
    for kind in ("dip", "momentum"):
        sub = [t for t in closed if t["type"] == kind]
        wins = [t for t in sub if t["pnl"] > 0]
        stats[kind] = {
            "n": len(sub),
            "win_rate": round(100 * len(wins) / len(sub), 1) if sub else None,
            "total_pnl": round(sum(t["pnl"] for t in sub), 2),
        }
    open_n = len([t for t in trades if t["pnl"] is None])
    if not quiet:
        print(json.dumps({"open": open_n, **stats}, indent=2))
    return {"open": open_n, **stats}


# ----------------------------------------------------------------------------
# HTML report
# ----------------------------------------------------------------------------

def render_html(ctx):
    """Self-contained dashboard.html — no external requests, opens anywhere."""

    def card(s, show_status=True):
        v = s["verdict"]
        cls = {"DIP": "dip", "MOMENTUM": "mom", "SHORT": "sht",
               "NEUTRAL": "neu", "AVOID": "avd"}[v]
        chg = s["day_chg"]
        chg_cls = "up" if chg >= 0 else "dn"
        status = s.get("status", "")
        status_html = ""
        if show_status and status and status != "—":
            st_cls = {"CONFIRMED": "st-conf", "PROVISIONAL": "st-prov",
                      "FADED": "st-fade", "NEW (1h)": "st-new",
                      "OFF-SCHEDULE": "st-off"}.get(status, "st-off")
            status_html = f'<span class="status {st_cls}">{status}</span>'
        levels = ""
        if s["sl"] and s["tp"]:
            levels = (f'<div class="levels"><span>SL <b>{s["sl"]}</b></span>'
                      f'<span>TP <b>{s["tp"]}</b></span>'
                      f'<span>ATR <b>{s["atr"]}</b></span></div>')
        rsi_txt = s["rsi"] if s["rsi"] is not None else "–"
        return f"""
        <div class="card {cls}">
          <div class="card-top">
            <span class="tk">{s["ticker"]}</span>
            <span class="verdict">{v}</span>{status_html}
          </div>
          <div class="px-row">
            <span class="px">{s["price"]}</span>
            <span class="chg {chg_cls}">{chg:+.2f}%</span>
            <span class="rsi">RSI {rsi_txt}</span>
          </div>
          {levels}
          <p class="why">{s["reason"]}</p>
        </div>"""

    def opp_card(s):
        dcls = {"LONG": "long", "SHORT": "short"}.get(s["direction"], "watch")
        return f"""
        <div class="card opp">
          <div class="card-top">
            <span class="tk">{s["ticker"]}</span>
            <span class="dir {dcls}">{s["direction"]}</span>
            <span class="score">score {s["score"]}</span>
          </div>
          <div class="px-row">
            <span class="px">{s["price"]}</span>
            <span class="chg {'up' if s['day_chg']>=0 else 'dn'}">{s["day_chg"]:+.2f}%</span>
            <span class="rsi">RSI {s["rsi"] if s["rsi"] is not None else "–"}</span>
            <span class="rsi">vol ×{s["vol_ratio"]}</span>
          </div>
          <div class="levels">
            <span>Entry <b>{s["price"]}</b></span>
            <span>SL <b>{s["sl"]}</b></span>
            <span>TP <b>{s["tp"]}</b></span>
          </div>
          <p class="why">{s["reason"]}</p>
        </div>"""

    eu_cards = "".join(card(s) for s in ctx["eu"]) or '<p class="empty">No EU data this run.</p>'
    us_cards = "".join(card(s) for s in ctx["us"]) or '<p class="empty">No US data this run.</p>'
    opp_cards = "".join(opp_card(s) for s in ctx["opps"]) or \
        '<p class="empty">Nothing in the universe is moving more than 1% today — quiet tape, no forced trades.</p>'
    cfd_cards = "".join(card(s, show_status=False) for s in ctx["cfd"]) or \
        '<p class="empty">No CFD data this run.</p>'

    js = ctx["journal"]
    def jline(kind):
        d = js[kind]
        wr = f'{d["win_rate"]}%' if d["win_rate"] is not None else "–"
        return (f'<div class="jstat"><span class="jlabel">{kind.upper()}</span>'
                f'<span>{d["n"]} closed</span><span>win {wr}</span>'
                f'<span class="{"up" if d["total_pnl"]>=0 else "dn"}">{d["total_pnl"]:+}</span></div>')

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Signal Desk — {ctx["stamp"]}</title>
<style>
  :root {{
    --bg:#0d1117; --panel:#141b26; --line:#243044;
    --ink:#dbe4f0; --dim:#7d8da3;
    --dip:#e8b339; --mom:#3fd0a4; --avd:#e05f5f; --neu:#5d6b80;
    --accent:#6ea8fe;
  }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--ink);
         font:14px/1.5 "Segoe UI",system-ui,sans-serif; padding:28px clamp(14px,4vw,48px); }}
  .mono {{ font-family:Consolas,"Cascadia Mono",monospace; }}
  header {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:14px;
            border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:24px; }}
  h1 {{ font-size:20px; letter-spacing:.04em; }}
  h1 span {{ color:var(--accent); }}
  .wave {{ font-family:Consolas,monospace; color:var(--dim); font-size:13px; }}
  h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.14em;
        color:var(--dim); margin:28px 0 12px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:12px; }}
  .card {{ background:var(--panel); border:1px solid var(--line);
           border-left:3px solid var(--neu); border-radius:8px; padding:12px 14px; }}
  .card.dip {{ border-left-color:var(--dip); }}
  .card.mom {{ border-left-color:var(--mom); }}
  .card.avd {{ border-left-color:var(--avd); }}
  .card.sht {{ border-left-color:#c77dff; }}
  .card.opp {{ border-left-color:var(--accent); }}
  .card-top {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
  .tk {{ font-family:Consolas,monospace; font-weight:700; font-size:15px; }}
  .verdict {{ font-size:11px; letter-spacing:.1em; color:var(--dim); }}
  .dip .verdict {{ color:var(--dip); }} .mom .verdict {{ color:var(--mom); }}
  .avd .verdict {{ color:var(--avd); }} .sht .verdict {{ color:#c77dff; }}
  .status {{ margin-left:auto; font-size:10px; letter-spacing:.08em;
             padding:2px 7px; border-radius:10px; border:1px solid var(--line); }}
  .st-conf {{ color:var(--mom); border-color:var(--mom); }}
  .st-prov {{ color:var(--dip); border-color:var(--dip); }}
  .st-fade {{ color:var(--dim); text-decoration:line-through; }}
  .st-new  {{ color:var(--accent); border-color:var(--accent); }}
  .st-off  {{ color:var(--dim); }}
  .dir {{ font-size:11px; letter-spacing:.1em; padding:2px 7px; border-radius:10px; }}
  .dir.long {{ color:var(--mom); border:1px solid var(--mom); }}
  .dir.watch {{ color:var(--dim); border:1px solid var(--line); }}
  .dir.short {{ color:#c77dff; border:1px solid #c77dff; }}
  .score {{ margin-left:auto; font-family:Consolas,monospace; font-size:12px; color:var(--accent); }}
  .px-row {{ display:flex; gap:12px; align-items:baseline; font-family:Consolas,monospace; }}
  .px {{ font-size:18px; font-weight:600; }}
  .chg.up {{ color:var(--mom); }} .chg.dn {{ color:var(--avd); }}
  .up {{ color:var(--mom); }} .dn {{ color:var(--avd); }}
  .rsi {{ color:var(--dim); font-size:12px; }}
  .levels {{ display:flex; gap:14px; margin-top:6px; font-family:Consolas,monospace;
             font-size:12px; color:var(--dim); }}
  .levels b {{ color:var(--ink); }}
  .why {{ margin-top:8px; font-size:12.5px; color:var(--dim); }}
  .empty {{ color:var(--dim); font-style:italic; }}
  .journal {{ display:flex; gap:24px; flex-wrap:wrap; background:var(--panel);
              border:1px solid var(--line); border-radius:8px; padding:12px 16px; }}
  .jstat {{ display:flex; gap:12px; font-family:Consolas,monospace; font-size:13px; }}
  .jlabel {{ color:var(--accent); letter-spacing:.08em; }}
  footer {{ margin-top:32px; border-top:1px solid var(--line); padding-top:12px;
            font-size:12px; color:var(--dim); }}
</style></head><body>
<header>
  <h1>SIGNAL<span>DESK</span></h1>
  <span class="wave">{ctx["wave_label"]} · generated {ctx["stamp"]} CET · data: Yahoo Finance</span>
</header>

<h2>EU Session — held</h2>
<div class="grid">{eu_cards}</div>

<h2>US Session — held</h2>
<div class="grid">{us_cards}</div>

<h2>Daily Opportunities — liquid non-held large caps moving today</h2>
<div class="grid">{opp_cards}</div>

<h2>CFD Picks — top 3 by tradability (demo account)</h2>
<div class="grid">{cfd_cards}</div>

<h2>Trade Journal</h2>
<div class="journal">
  <div class="jstat"><span class="jlabel">OPEN</span><span>{js["open"]} positions</span></div>
  {jline("dip")}
  {jline("momentum")}
</div>

<footer>Stops 1.5×ATR · targets 2.5×ATR (inverted for shorts) · two-wave confirmation filters opening noise.
Decision aid, not advice — verify on XTB before acting.{ctx["note"]}</footer>
</body></html>"""
    DASHBOARD_FILE.write_text(html, encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_scan(args):
    t = now_cet()
    wave = args.wave if args.wave != "auto" else detect_wave(t)
    state = load_json(STATE_FILE, {})
    today = t.strftime("%Y-%m-%d")
    note = ""

    # Decide what to scan live vs reuse from persisted state
    scan_eu = scan_us = True
    if wave is None:                                   # off-schedule run
        scan_eu = market_open_now("EU", t)
        scan_us = market_open_now("US", t)
        reused = []
        if not scan_eu: reused.append("EU")
        if not scan_us: reused.append("US")
        if reused:
            note = (f" Off-schedule run at {t.strftime('%H:%M')}: "
                    f"{'/'.join(reused)} market closed — reused last scan from state.")
        print(f"Off-schedule run. Live scan — EU: {scan_eu}, US: {scan_us}")
    else:
        scan_eu = "EU" in WAVES[wave]["markets"]
        scan_us = "US" in WAVES[wave]["markets"]
        print(f"Wave: {WAVES[wave]['label']}")

    eu_signals, us_signals = [], []

    if scan_eu:
        print("Scanning EU watchlist...")
        hist = fetch_history(WATCHLIST_EU, demo=args.demo)
        eu_signals = [analyze(tk, h) for tk, h in hist.items()]
        eu_signals = apply_two_wave(eu_signals, wave if wave in ("eu_open", "eu_conf") else None, state)
        state.setdefault(today, {})["eu_last"] = eu_signals
    else:
        eu_signals = state.get(today, {}).get("eu_last", [])
        if not eu_signals:  # fall back to most recent prior day
            for day in sorted(state.keys(), reverse=True):
                if state[day].get("eu_last"):
                    eu_signals = state[day]["eu_last"]
                    break

    if scan_us:
        print("Scanning US watchlist...")
        hist = fetch_history(WATCHLIST_US, demo=args.demo)
        us_signals = [analyze(tk, h) for tk, h in hist.items()]
        us_signals = apply_two_wave(us_signals, wave if wave in ("us_open", "us_conf") else None, state)
        state.setdefault(today, {})["us_last"] = us_signals
    else:
        us_signals = state.get(today, {}).get("us_last", [])
        if not us_signals:
            for day in sorted(state.keys(), reverse=True):
                if state[day].get("us_last"):
                    us_signals = state[day]["us_last"]
                    break

    # Daily Opportunities (run whenever US is scannable — it's a US universe)
    opps = []
    if scan_us or args.demo:
        print("Scanning Daily Opportunities universe "
              f"({len(OPPORTUNITY_UNIVERSE)} names)...")
        hist = fetch_history(OPPORTUNITY_UNIVERSE, demo=args.demo)
        opps = scan_opportunities(hist, held=set(WATCHLIST))
        state.setdefault(today, {})["opps_last"] = opps
    else:
        opps = state.get(today, {}).get("opps_last", [])
        note += " Opportunities reused from last scan (US closed)."

    # CFD picks
    print(f"Scoring CFD universe ({len(CFD_UNIVERSE)} names)...")
    hist = fetch_history(CFD_UNIVERSE, demo=args.demo)
    cfd = scan_cfd(hist)

    # Prune state older than 7 days
    cutoff = (t - timedelta(days=7)).strftime("%Y-%m-%d")
    for day in [d for d in state if d < cutoff]:
        del state[day]
    save_json(STATE_FILE, state)

    ctx = {
        "eu": eu_signals, "us": us_signals, "opps": opps, "cfd": cfd,
        "journal": journal_stats(quiet=True),
        "wave_label": WAVES[wave]["label"] if wave else "Off-schedule run",
        "stamp": t.strftime("%Y-%m-%d %H:%M"),
        "note": note + (" DEMO DATA — synthetic prices for testing." if args.demo else ""),
    }
    render_html(ctx)
    print(f"\nDashboard written: {DASHBOARD_FILE}")

    confirmed = [s["ticker"] for s in eu_signals + us_signals
                 if s.get("status") == "CONFIRMED"]
    if confirmed:
        print(f"CONFIRMED signals: {', '.join(confirmed)}")
    if opps:
        print("Top opportunity: "
              f"{opps[0]['ticker']} ({opps[0]['direction']}, score {opps[0]['score']})")

    if not args.no_open:
        try:
            webbrowser.open(DASHBOARD_FILE.as_uri())
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="Market signal agent v5")
    sub = p.add_subparsers(dest="cmd")

    p.add_argument("--wave", choices=["auto", "eu_open", "eu_conf", "us_open", "us_conf"],
                   default="auto", help="Force a wave (default: auto-detect from CET time)")
    p.add_argument("--demo", action="store_true", help="Synthetic data, no network")
    p.add_argument("--no-open", action="store_true", help="Don't open the dashboard in browser")

    j = sub.add_parser("journal", help="Trade journal")
    jsub = j.add_subparsers(dest="jcmd", required=True)
    ja = jsub.add_parser("add")
    ja.add_argument("--ticker", required=True)
    ja.add_argument("--type", choices=["dip", "momentum"], required=True)
    ja.add_argument("--entry", type=float, required=True)
    ja.add_argument("--size", type=float, required=True)
    jc = jsub.add_parser("close")
    jc.add_argument("--id", type=int, required=True)
    jc.add_argument("--exit", type=float, required=True)
    jsub.add_parser("stats")

    args = p.parse_args()
    if args.cmd == "journal":
        {"add": journal_add, "close": journal_close,
         "stats": journal_stats}[args.jcmd](args)
    else:
        run_scan(args)


if __name__ == "__main__":
    main()
