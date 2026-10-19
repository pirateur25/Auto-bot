# backtest/engine.py
# Moteur de simulation bougie par bougie.
#
# Règles (volontairement prudentes pour ne pas obtenir de résultats trop beaux) :
#  - Le signal est calculé à la CLÔTURE de la bougie i ; l'ordre est exécuté à l'OUVERTURE de la bougie i+1.
#  - Frais sur chaque ordre (entrée et sortie) + glissement sur les entrées et les stop-loss.
#  - Si SL et TP sont touchés dans la même bougie : on suppose que le SL est touché en premier.
#  - Si une bougie s'ouvre au-delà du SL (gap), la sortie se fait à l'ouverture (donc pire que le SL).
#  - Taille de position = risque fixé (% du capital) / distance au SL, plafonnée par le levier maximum.
#  - Une seule position à la fois.

import numpy as np
import pandas as pd


def run_backtest(df, strategy, capital=10000.0, risk_pct=1.0, fee_pct=0.1, slippage_pct=0.02,
                 max_leverage=1.0, allow_short=True, max_bars=None):
    df = strategy.prepare(df.copy()).reset_index(drop=True)
    d = {col: df[col].to_numpy() for col in df.columns}
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    times = df["time"]
    n = len(df)
    fee = fee_pct / 100.0
    slip = slippage_pct / 100.0

    balance = float(capital)
    pos = None
    pending = None
    trades = []
    equity = np.full(n, np.nan)

    def close_position(i, price, reason):
        nonlocal balance, pos
        direction = 1 if pos["side"] == "long" else -1
        gross = (price - pos["entry"]) * pos["qty"] * direction
        exit_fee = pos["qty"] * price * fee
        balance += gross - exit_fee
        net = gross - pos["entry_fee"] - exit_fee
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": times.iloc[i], "side": pos["side"],
            "entry": pos["entry"], "exit": price, "sl": pos["sl"], "tp": pos["tp"], "qty": pos["qty"],
            "bars": i - pos["entry_i"], "reason": reason,
            "fees": pos["entry_fee"] + exit_fee, "pnl": net,
            "r": net / pos["risk_amount"] if pos["risk_amount"] > 0 else 0.0,
            "balance_after": balance,
        })
        pos = None

    for i in range(n):
        # 1) exécution de l'ordre décidé à la clôture précédente, à l'ouverture de cette bougie
        if pending is not None and pos is None:
            sig, pending = pending, None
            side = sig["side"]
            entry = o[i] * (1 + slip) if side == "long" else o[i] * (1 - slip)
            sl = float(sig["sl"])
            tp = sig.get("tp")
            risk_dist = abs(entry - sl)
            if tp is None and sig.get("rr") is not None and risk_dist > 0:
                tp = entry + sig["rr"] * risk_dist if side == "long" else entry - sig["rr"] * risk_dist
            valid = risk_dist > 0 and ((side == "long" and sl < entry) or (side == "short" and sl > entry))
            if valid and tp is not None:
                valid = (side == "long" and tp > entry) or (side == "short" and tp < entry)
            if valid and (side == "long" or allow_short):
                qty = (balance * risk_pct / 100.0) / risk_dist
                qty = min(qty, balance * max_leverage / entry)
                entry_fee = qty * entry * fee
                balance -= entry_fee
                pos = {"side": side, "entry": entry, "sl": sl, "tp": tp, "qty": qty, "entry_i": i,
                       "entry_time": times.iloc[i], "entry_fee": entry_fee, "risk_amount": qty * risk_dist}

        # 2) gestion de la position sur cette bougie
        if pos is not None:
            gap = i > pos["entry_i"]
            sl, tp = pos["sl"], pos["tp"]
            if pos["side"] == "long":
                if l[i] <= sl:
                    stop = min(sl, o[i]) if gap else sl
                    close_position(i, stop * (1 - slip), "SL")
                elif tp is not None and h[i] >= tp:
                    close_position(i, max(tp, o[i]) if gap else tp, "TP")
            else:
                if h[i] >= sl:
                    stop = max(sl, o[i]) if gap else sl
                    close_position(i, stop * (1 + slip), "SL")
                elif tp is not None and l[i] <= tp:
                    close_position(i, min(tp, o[i]) if gap else tp, "TP")
            if pos is not None and max_bars and i - pos["entry_i"] >= max_bars:
                close_position(i, c[i] * (1 - slip if pos["side"] == "long" else 1 + slip), "TIMEOUT")

        # 3) valeur du compte à la clôture
        eq = balance
        if pos is not None:
            direction = 1 if pos["side"] == "long" else -1
            eq += (c[i] - pos["entry"]) * pos["qty"] * direction
        equity[i] = eq

        # 4) nouveau signal à la clôture (exécuté à l'ouverture de la bougie suivante)
        if pos is None and pending is None and i < n - 1:
            sig = strategy.signal(d, i)
            if sig:
                pending = sig

    if pos is not None:  # position encore ouverte à la fin des données
        close_position(n - 1, c[-1], "END")
        equity[-1] = balance

    trades_df = pd.DataFrame(trades)
    equity_s = pd.Series(equity, index=times, name="equity")
    return trades_df, equity_s, compute_stats(trades_df, equity_s, df, capital)


def compute_stats(trades, equity, df, capital):
    final = float(equity.iloc[-1])
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min() * 100)
    buy_hold = float((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100)
    s = {
        "capital_start": capital, "capital_end": final,
        "return_pct": (final / capital - 1) * 100,
        "buy_hold_pct": buy_hold, "max_drawdown_pct": max_dd,
        "trades": len(trades),
    }
    if len(trades):
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        gross_win, gross_loss = wins["pnl"].sum(), -losses["pnl"].sum()
        s.update({
            "longs": int((trades["side"] == "long").sum()), "shorts": int((trades["side"] == "short").sum()),
            "win_rate_pct": len(wins) / len(trades) * 100,
            "avg_r": float(trades["r"].mean()),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_bars": float(trades["bars"].mean()),
            "fees_paid": float(trades["fees"].sum()),
            "best_r": float(trades["r"].max()), "worst_r": float(trades["r"].min()),
        })
    return s
  
