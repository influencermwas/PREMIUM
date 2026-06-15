import os
import json
import math
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}

PREMIUM_PRICE = int(os.getenv("PREMIUM_PRICE", "35"))
VIP_DAYS = int(os.getenv("VIP_DAYS", "30"))
DAILY_SIGNAL_LIMIT = int(os.getenv("DAILY_SIGNAL_LIMIT", "5"))

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "79"))
A_MIN = float(os.getenv("A_SIGNAL_MIN_CONFIDENCE", "85"))
APLUS_MIN = float(os.getenv("A_PLUS_MIN_CONFIDENCE", "90"))

DATA_FILE = Path("vip_users.json")
USAGE_FILE = Path("daily_usage.json")
CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

# You can add/remove symbols here.
SCAN_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT",
    "LTCUSDT", "TRXUSDT", "DOTUSDT", "NEARUSDT", "INJUSDT",
]

TIMEFRAME = os.getenv("TIMEFRAME", "15m")
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "150"))


# -------------------------
# Storage
# -------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def today_key() -> str:
    return now_utc().strftime("%Y-%m-%d")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_vips() -> dict:
    return load_json(DATA_FILE)


def save_vips(data: dict) -> None:
    save_json(DATA_FILE, data)


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def is_vip(user_id: int) -> bool:
    data = load_vips()
    user = data.get(str(user_id))
    if not user:
        return False
    try:
        expiry = datetime.fromisoformat(user["expires_at"])
    except Exception:
        return False
    return expiry > now_utc()


def vip_status_text(user_id: int) -> str:
    data = load_vips()
    user = data.get(str(user_id))
    if not user:
        return "❌ You are not VIP yet."

    expiry = datetime.fromisoformat(user["expires_at"])
    if expiry <= now_utc():
        return f"❌ Your VIP expired on {expiry.strftime('%Y-%m-%d %H:%M UTC')}."

    remaining = expiry - now_utc()
    return (
        "✅ You are VIP.\n"
        f"Expires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Remaining: {remaining.days} days {remaining.seconds // 3600} hours"
    )


def add_vip_user(user_id: int, days: int, added_by: int = 0) -> datetime:
    data = load_vips()
    current = data.get(str(user_id))

    if current:
        old_expiry = datetime.fromisoformat(current["expires_at"])
        base_time = old_expiry if old_expiry > now_utc() else now_utc()
    else:
        base_time = now_utc()

    expiry = base_time + timedelta(days=days)
    data[str(user_id)] = {
        "user_id": user_id,
        "expires_at": expiry.isoformat(),
        "added_by": added_by,
        "updated_at": now_utc().isoformat(),
    }
    save_vips(data)
    return expiry


def user_usage_left(user_id: int) -> Tuple[int, int]:
    usage = load_json(USAGE_FILE)
    day = today_key()
    user_day = usage.get(str(user_id), {}).get(day, 0)
    return max(0, DAILY_SIGNAL_LIMIT - user_day), user_day


def increment_usage(user_id: int) -> None:
    usage = load_json(USAGE_FILE)
    day = today_key()
    usage.setdefault(str(user_id), {})
    usage[str(user_id)][day] = usage[str(user_id)].get(day, 0) + 1
    save_json(USAGE_FILE, usage)


# -------------------------
# Market data + indicators
# -------------------------

def fetch_binance_klines(symbol: str, interval: str = TIMEFRAME, limit: int = KLINE_LIMIT) -> Optional[pd.DataFrame]:
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
        r.raise_for_status()
        rows = r.json()
        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df[["time", "open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        logging.warning("Failed fetching %s: %s", symbol, e)
        return None


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_recent_swings(df: pd.DataFrame, lookback: int = 40) -> Tuple[float, float]:
    recent = df.tail(lookback)
    return float(recent["low"].min()), float(recent["high"].max())


def confidence_grade(conf: float) -> str:
    if conf >= APLUS_MIN:
        return "A+"
    if conf >= A_MIN:
        return "A"
    if conf >= MIN_CONFIDENCE:
        return "B"
    return "LOW"


def analyze_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    df = fetch_binance_klines(symbol)
    if df is None or len(df) < 80:
        return None

    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    df["vol_ma"] = df["volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(last["close"])
    atr_val = float(last["atr"]) if not math.isnan(last["atr"]) else price * 0.01
    rsi_val = float(last["rsi"]) if not math.isnan(last["rsi"]) else 50
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    vol_boost = float(last["volume"] / last["vol_ma"]) if last["vol_ma"] and not math.isnan(last["vol_ma"]) else 1.0
    swing_low, swing_high = find_recent_swings(df)

    bullish = price > ema20 > ema50
    bearish = price < ema20 < ema50

    # Pullback/retest logic
    near_ema20 = abs(price - ema20) <= atr_val * 0.8
    breakout_up = price > swing_high - atr_val * 0.4
    breakout_down = price < swing_low + atr_val * 0.4

    confidence = 60
    reasons = []

    if bullish:
        confidence += 12
        reasons.append("EMA20 is above EMA50 and price is trading above both averages.")
    if bearish:
        confidence += 12
        reasons.append("EMA20 is below EMA50 and price is trading below both averages.")

    if near_ema20:
        confidence += 6
        reasons.append("Price is near EMA20, giving a clean retest/pullback area.")

    if 45 <= rsi_val <= 68 and bullish:
        confidence += 7
        reasons.append("RSI is healthy for bullish continuation, not extremely overbought.")
    elif 32 <= rsi_val <= 55 and bearish:
        confidence += 7
        reasons.append("RSI is healthy for bearish continuation, not extremely oversold.")

    if breakout_up and bullish:
        confidence += 5
        reasons.append("Price is pressing recent resistance, showing breakout pressure.")
    if breakout_down and bearish:
        confidence += 5
        reasons.append("Price is pressing recent support, showing breakdown pressure.")

    if vol_boost >= 1.2:
        confidence += 6
        reasons.append("Volume is above average, supporting momentum.")

    # Decide direction
    if bullish:
        direction = "BUY"
        entry = price
        # Wide SL: below swing/EMA area with ATR buffer, not too close
        sl = min(swing_low, ema50) - atr_val * 0.8
        risk = abs(entry - sl)
        tp1 = entry + risk * 1.2
        tp2 = entry + risk * 2.0
        tp3 = entry + risk * 3.0
        tp4 = entry + risk * 4.0
        setup = "Bullish trend continuation. Price is holding above EMA20/EMA50 with a possible pullback or breakout continuation setup."
        invalidation = "Invalid if price closes below the protected support zone and SL area."
    elif bearish:
        direction = "SELL"
        entry = price
        sl = max(swing_high, ema50) + atr_val * 0.8
        risk = abs(sl - entry)
        tp1 = entry - risk * 1.2
        tp2 = entry - risk * 2.0
        tp3 = entry - risk * 3.0
        tp4 = entry - risk * 4.0
        setup = "Bearish trend continuation. Price is holding below EMA20/EMA50 with rejection pressure from the trend area."
        invalidation = "Invalid if price closes above the protected resistance zone and SL area."
    else:
        return None

    if confidence < MIN_CONFIDENCE:
        return None

    return {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp4": tp4,
        "confidence": min(96, round(confidence, 1)),
        "grade": confidence_grade(min(96, confidence)),
        "rsi": rsi_val,
        "atr": atr_val,
        "volume_boost": vol_boost,
        "setup": setup,
        "reason": " ".join(reasons),
        "invalidation": invalidation,
        "df": df,
    }


def find_best_signal() -> Optional[Dict[str, Any]]:
    best = None
    for symbol in SCAN_SYMBOLS:
        signal = analyze_symbol(symbol)
        if signal and (best is None or signal["confidence"] > best["confidence"]):
            best = signal
    return best


# -------------------------
# Premium message + chart
# -------------------------

def fmt_price(x: float) -> str:
    if x >= 100:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:.8f}"


def rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    if risk == 0:
        return 0
    return abs(tp - entry) / risk


def build_signal_text(s: Dict[str, Any]) -> str:
    emoji = "🟢" if s["direction"] == "BUY" else "🔴"
    return (
        f"🔥 <b>PREMIUM VIP SIGNAL</b> 🔥\n\n"
        f"{emoji} <b>{s['symbol']}</b>\n"
        f"Direction: <b>{s['direction']}</b>\n"
        f"Timeframe: <b>{s['timeframe']}</b>\n"
        f"Signal Grade: <b>{s['grade']}</b>\n"
        f"Confidence: <b>{s['confidence']}%</b>\n\n"
        f"📌 <b>ENTRY ZONE</b>\n"
        f"Entry: <code>{fmt_price(s['entry'])}</code>\n"
        f"Entry Style: Wait for candle confirmation around entry zone.\n\n"
        f"🛡 <b>STOP LOSS</b>\n"
        f"SL: <code>{fmt_price(s['sl'])}</code>\n"
        f"SL Type: Wide protective SL, placed beyond market noise/wicks.\n\n"
        f"🎯 <b>TAKE PROFITS</b>\n"
        f"TP1: <code>{fmt_price(s['tp1'])}</code> | RR: {rr(s['entry'], s['sl'], s['tp1']):.2f}R\n"
        f"TP2: <code>{fmt_price(s['tp2'])}</code> | RR: {rr(s['entry'], s['sl'], s['tp2']):.2f}R\n"
        f"TP3: <code>{fmt_price(s['tp3'])}</code> | RR: {rr(s['entry'], s['sl'], s['tp3']):.2f}R\n"
        f"TP4: <code>{fmt_price(s['tp4'])}</code> | RR: {rr(s['entry'], s['sl'], s['tp4']):.2f}R\n\n"
        f"📊 <b>CHART SETUP</b>\n"
        f"{s['setup']}\n\n"
        f"🧠 <b>WHY THIS SIGNAL?</b>\n"
        f"{s['reason']}\n"
        f"RSI: {s['rsi']:.1f} | ATR buffer used for SL | Volume strength: {s['volume_boost']:.2f}x\n\n"
        f"❌ <b>INVALIDATION</b>\n"
        f"{s['invalidation']}\n\n"
        f"⚠️ <b>TRADE MANAGEMENT</b>\n"
        f"• Use low risk per trade.\n"
        f"• Take partial profit at TP1.\n"
        f"• Move SL to breakeven after TP1 if momentum continues.\n"
        f"• TP3/TP4 are runner targets; do not close too early if trend is strong.\n\n"
        f"⏰ Sent: {now_utc().strftime('%Y-%m-%d %H:%M UTC')}"
    )


def create_chart_image(s: Dict[str, Any]) -> Path:
    df = s["df"].tail(60).copy()
    x = range(len(df))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x, df["close"], linewidth=2, label="Close")
    ax.plot(x, df["ema20"], linewidth=1.5, label="EMA20")
    ax.plot(x, df["ema50"], linewidth=1.5, label="EMA50")

    levels = [
        ("ENTRY", s["entry"], "--"),
        ("SL", s["sl"], "-."),
        ("TP1", s["tp1"], ":"),
        ("TP2", s["tp2"], ":"),
        ("TP3", s["tp3"], ":"),
        ("TP4", s["tp4"], ":"),
    ]

    for name, price, style in levels:
        ax.axhline(price, linestyle=style, linewidth=1.8)
        ax.text(len(df) + 0.5, price, f"{name} {fmt_price(price)}", va="center", fontsize=10)

    ax.set_title(
        f"{s['symbol']} {s['direction']} Premium Setup | {s['timeframe']} | {s['grade']} {s['confidence']}%",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("Recent candles")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.02,
        "Auto-analysis: trend • EMA structure • RSI • ATR SL buffer • volume",
        transform=ax.transAxes,
        fontsize=10,
        alpha=0.8,
    )

    path = CHART_DIR / f"{s['symbol']}_{s['direction']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# -------------------------
# Telegram handlers
# -------------------------

def main_menu(user_id: int) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("🔥 Get Premium Signal", callback_data="premium_signal")],
        [InlineKeyboardButton("💎 Get VIP", callback_data="get_vip")],
        [InlineKeyboardButton("📊 VIP Status", callback_data="status")],
    ]
    if is_admin(user_id):
        buttons.append([InlineKeyboardButton("✅ Admin: Activate VIP", callback_data="admin_help")])
    return InlineKeyboardMarkup(buttons)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔥 <b>Auto Premium VIP Signals Bot</b>\n\n"
        "Paid users tap <b>Get Premium Signal</b> and the bot automatically scans the market, analyzes setups, and sends the best premium signal with chart, TP and SL.\n\n"
        "No admin signal typing needed.",
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu(update.effective_user.id),
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    left, used = user_usage_left(update.effective_user.id)
    await update.message.reply_text(
        vip_status_text(update.effective_user.id) + f"\n\nDaily signals used: {used}/{DAILY_SIGNAL_LIMIT}\nRemaining today: {left}"
    )


async def get_vip_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text(
        f"💎 <b>Get VIP Access</b>\n\n"
        f"Price: Ksh {PREMIUM_PRICE}\n"
        f"After payment, send this Telegram ID to admin:\n"
        f"<code>{user_id}</code>\n\n"
        f"Admin activates using:\n"
        f"<code>/addvip {user_id} {VIP_DAYS}</code>",
        parse_mode=ParseMode.HTML,
    )


async def premium_signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_vip(user_id):
        return await update.message.reply_text(
            "❌ VIP only.\n\nUse /get_vip to activate premium access."
        )

    left, used = user_usage_left(user_id)
    if left <= 0:
        return await update.message.reply_text(
            f"⛔ Daily premium signal limit reached.\nUsed: {used}/{DAILY_SIGNAL_LIMIT}\nTry again tomorrow."
        )

    msg = await update.message.reply_text("🔎 Scanning market for the best premium setup...")

    signal = find_best_signal()
    if not signal:
        return await msg.edit_text(
            "⚠️ No strong premium setup found right now.\n\nMarket is not clean enough for VIP entry. Try again later."
        )

    text = build_signal_text(signal)
    chart = create_chart_image(signal)
    increment_usage(user_id)

    with chart.open("rb") as photo:
        await update.message.reply_photo(
            photo=photo,
            caption=text,
            parse_mode=ParseMode.HTML,
        )

    left_after, used_after = user_usage_left(user_id)
    await msg.edit_text(f"✅ Premium signal delivered.\nRemaining today: {left_after}/{DAILY_SIGNAL_LIMIT}")


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    user_id = q.from_user.id

    if q.data == "premium_signal":
        if not is_vip(user_id):
            return await q.message.reply_text("❌ VIP only. Tap Get VIP to activate.")
        left, used = user_usage_left(user_id)
        if left <= 0:
            return await q.message.reply_text(f"⛔ Daily limit reached. Used: {used}/{DAILY_SIGNAL_LIMIT}")

        wait = await q.message.reply_text("🔎 Scanning market for the best premium setup...")
        signal = find_best_signal()
        if not signal:
            return await wait.edit_text("⚠️ No strong premium setup found right now. Try again later.")

        text = build_signal_text(signal)
        chart = create_chart_image(signal)
        increment_usage(user_id)

        with chart.open("rb") as photo:
            await q.message.reply_photo(photo=photo, caption=text, parse_mode=ParseMode.HTML)

        left_after, _ = user_usage_left(user_id)
        await wait.edit_text(f"✅ Premium signal delivered.\nRemaining today: {left_after}/{DAILY_SIGNAL_LIMIT}")

    elif q.data == "get_vip":
        await get_vip_cmd(update, context)

    elif q.data == "status":
        left, used = user_usage_left(user_id)
        await q.message.reply_text(
            vip_status_text(user_id) + f"\n\nDaily signals used: {used}/{DAILY_SIGNAL_LIMIT}\nRemaining today: {left}"
        )

    elif q.data == "admin_help":
        if not is_admin(user_id):
            return await q.message.reply_text("❌ Admin only.")
        await q.message.reply_text(
            "Admin commands:\n"
            "/addvip user_id days\n"
            "/removevip user_id\n"
            "/vips\n\n"
            "Signals are automatic. Admin does not type signals."
        )


async def addvip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("❌ Admin only.")
    if len(context.args) != 2:
        return await update.message.reply_text("Usage: /addvip user_id days")
    try:
        user_id = int(context.args[0])
        days = int(context.args[1])
    except ValueError:
        return await update.message.reply_text("User ID and days must be numbers.")

    expiry = add_vip_user(user_id, days, update.effective_user.id)
    await update.message.reply_text(
        f"✅ VIP activated.\nUser: {user_id}\nExpires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    try:
        await context.bot.send_message(
            chat_id=user_id,
            text=f"✅ Your VIP is active.\nYou can now tap 🔥 Get Premium Signal.\nExpires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}",
        )
    except Exception:
        pass


async def removevip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("❌ Admin only.")
    if len(context.args) != 1:
        return await update.message.reply_text("Usage: /removevip user_id")

    data = load_vips()
    uid = context.args[0]
    if uid not in data:
        return await update.message.reply_text("User is not VIP.")
    data.pop(uid)
    save_vips(data)
    await update.message.reply_text(f"✅ VIP removed for {uid}.")


async def vips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return await update.message.reply_text("❌ Admin only.")
    data = load_vips()
    if not data:
        return await update.message.reply_text("No VIP users yet.")

    lines = ["💎 VIP users:"]
    for uid, info in data.items():
        expiry = datetime.fromisoformat(info["expires_at"])
        mark = "✅" if expiry > now_utc() else "⛔"
        lines.append(f"{mark} {uid} - {expiry.strftime('%Y-%m-%d %H:%M UTC')}")
    await update.message.reply_text("\n".join(lines))


def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("get_vip", get_vip_cmd))
    app.add_handler(CommandHandler("premium_signal", premium_signal_cmd))

    app.add_handler(CommandHandler("addvip", addvip))
    app.add_handler(CommandHandler("removevip", removevip))
    app.add_handler(CommandHandler("vips", vips))

    app.add_handler(CallbackQueryHandler(button_handler))

    app.run_polling()


if __name__ == "__main__":
    main()
