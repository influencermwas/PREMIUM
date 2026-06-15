# Auto VIP Premium Signals Bot

This version does **not** need admin to type signals.

Paid/VIP user taps **Get Premium Signal**, then the bot:

1. Scans crypto markets from Binance public data
2. Calculates EMA20, EMA50, RSI, ATR, volume strength and swing levels
3. Selects the best high-confidence setup
4. Sends a premium signal with:
   - Entry
   - Wide protective SL
   - TP1, TP2, TP3, TP4
   - Risk/reward
   - Reason
   - Invalidation
   - Trade management
   - Chart setup image

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
export BOT_TOKEN="YOUR_BOT_TOKEN"
export ADMIN_IDS="YOUR_TELEGRAM_ID"
python app.py
```

Windows PowerShell:

```powershell
$env:BOT_TOKEN="YOUR_BOT_TOKEN"
$env:ADMIN_IDS="YOUR_TELEGRAM_ID"
python app.py
```

## User flow

User opens bot and taps:

```text
🔥 Get Premium Signal
```

If VIP is active, the bot automatically analyzes and sends a premium signal.

## Admin commands

```text
/addvip user_id days
/removevip user_id
/vips
```

## User commands

```text
/start
/status
/get_vip
/premium_signal
```

## Settings

Edit `.env.example` or environment variables:

```text
PREMIUM_PRICE=35
VIP_DAYS=30
DAILY_SIGNAL_LIMIT=5
MIN_CONFIDENCE=79
A_SIGNAL_MIN_CONFIDENCE=85
A_PLUS_MIN_CONFIDENCE=90
TIMEFRAME=15m
```

## Important

This is a signal/analysis bot, not guaranteed profit. Always use risk management.
