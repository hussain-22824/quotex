"""
2-Minute Signal Scanner (real-market forex pairs, 1-minute candles).
Shows CALL (up) / PUT (down) setups for a 2-minute expiry, and the
backtested win rate of the same rules on the last ~5 days, compared with
the break-even win rate for your payout.

Real market data only. It does NOT cover broker OTC pairs.
Not financial advice. Short-expiry trading is very high risk.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="2-Min Signals", page_icon="⏱️", layout="centered")

PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}
EXPIRY_BARS = 2  # 2 one-minute candles = 2 minutes


def add_signals(df):
    c = df["Close"]
    e9 = c.ewm(span=9, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()

    call_raw = (e9 > e21) & (e21 > e50) & rsi.between(45, 65) & (hist > 0) & (hist > hist.shift())
    put_raw = (e9 < e21) & (e21 < e50) & rsi.between(35, 55) & (hist < 0) & (hist < hist.shift())
    df = df.copy()
    df["call"] = call_raw & ~call_raw.shift(1, fill_value=False)
    df["put"] = put_raw & ~put_raw.shift(1, fill_value=False)
    df["future"] = c.shift(-EXPIRY_BARS)
    return df


def backtest(df):
    wins = losses = 0
    for side in ("call", "put"):
        sub = df[df[side] & df["future"].notna()]
        if side == "call":
            wins += int((sub["future"] > sub["Close"]).sum())
            losses += int((sub["future"] < sub["Close"]).sum())
        else:
            wins += int((sub["future"] < sub["Close"]).sum())
            losses += int((sub["future"] > sub["Close"]).sum())
    return wins, losses


@st.cache_data(ttl=45, show_spinner=False)
def download(symbols):
    return yf.download(list(symbols), period="5d", interval="1m", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)


st.title("⏱️ 2-Minute Signals")
st.caption("Real-market forex pairs, 1-minute candles, 2-minute expiry.")

with st.expander("Settings"):
    payout = st.number_input("Broker payout (%)", min_value=50, max_value=95, value=85)
breakeven = 100 / (100 + payout) * 100

if st.button("🔍 SCAN NOW", type="primary", use_container_width=True):
    st.cache_data.clear()

try:
    data = download(tuple(PAIRS.values()))
except Exception:
    st.error("Could not load data. Wait a minute and press SCAN NOW.")
    st.stop()

rows = []
for name, sym in PAIRS.items():
    try:
        df = data[sym].dropna()
        if len(df) < 120:
            continue
        df = add_signals(df)
        w, l = backtest(df)
        n = w + l
        rate = w / n * 100 if n else 0.0
        recent = df.tail(3)  # signal must be at most ~2 minutes old
        if recent["call"].any():
            sig = "🟢 CALL (up)"
            age = len(recent) - 1 - int(np.where(recent["call"].values)[0][-1])
        elif recent["put"].any():
            sig = "🔴 PUT (down)"
            age = len(recent) - 1 - int(np.where(recent["put"].values)[0][-1])
        else:
            sig, age = "WAIT", None
        if n < 30:
            verdict = "Too few trades"
        elif rate >= breakeven + 3:
            verdict = "Above break-even"
        elif rate >= breakeven:
            verdict = "Barely break-even"
        else:
            verdict = "Below break-even"
        rows.append({
            "Pair": name, "Signal": sig,
            "Signal age (min)": age if age is not None else "-",
            "Price": round(float(df["Close"].iloc[-1]), 5),
            "Past win rate %": round(rate, 1), "Past trades": n, "Verdict": verdict,
        })
    except Exception:
        continue

if not rows:
    st.info("No data. Forex is closed from Friday evening to Sunday evening (New York time).")
    st.stop()

out = pd.DataFrame(rows)
live = out[out["Signal"] != "WAIT"]
if live.empty:
    st.info("No signals right now. WAIT and press SCAN NOW again in a minute.")
else:
    st.success(f"{len(live)} signal(s) right now")
    st.dataframe(live, use_container_width=True, hide_index=True)

with st.expander("All pairs and past results"):
    st.dataframe(out.sort_values("Past win rate %", ascending=False),
                 use_container_width=True, hide_index=True)

st.warning(f"With {payout}% payout you must win more than {breakeven:.1f}% of trades just to "
           "break even. Past results above are from the last ~5 days only, ignore spreads and "
           "price delay, and do not guarantee future results. Prices here come from Yahoo and "
           "can differ from your broker's, so a signal may already be stale. "
           "Only trade money you can afford to lose. Not financial advice.")
