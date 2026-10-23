# backtest/run_backtest.py
# Lance la simulation pour une ou plusieurs paires, écrit les résultats dans results/
# et envoie un résumé (et la courbe de capital) sur Telegram si BOT_TOKEN / CHAT_ID sont définis.
#
# Exemple :
#   python backtest/run_backtest.py --symbols BTCUSDT,ETHUSDT --interval 1h --start 2022-01-01

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import engine  # noqa: E402
import strategy  # noqa: E402


def apply_params(text):
    """'k=v;k2=v2' -> modifie strategy.PARAMS (chaque valeur est convertie dans le type de la valeur par défaut)."""
    changed = {}
    for part in [x.strip() for x in (text or "").replace("\n", ";").split(";") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"réglage invalide '{part}' (format attendu : nom=valeur)")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in strategy.PARAMS:
            raise ValueError(f"réglage inconnu '{k}'. Possibles : {', '.join(strategy.PARAMS)}")
        cur = strategy.PARAMS[k]
        strategy.PARAMS[k] = (float(v) if isinstance(cur, float) else int(float(v)) if isinstance(cur, int)
                              else v.replace(" ", ""))
        changed[k] = strategy.PARAMS[k]
    return changed


def setup_breakdown(trades):
    """Résultats par type de setup (ex. bubble5m|4h = bulle 5m, range H4) : n, % gagnants, espérance en R."""
    if not len(trades) or "setup" not in trades:
        return ""
    t = trades.copy()
    t["kind"] = t["setup"].astype(str).str.replace(r"[YR]$", "", regex=True)
    rows = []
    for k, g in t.groupby("kind"):
        rows.append((len(g), f"  {k}: {len(g)} trades | gagnants {(g['pnl'] > 0).mean() * 100:.0f} % | {g['r'].mean():+.2f} R"))
    rows.sort(reverse=True)
    return "Par setup :\n" + "\n".join(r for _, r in rows[:12])


def fmt_summary(symbol, interval, source, start, end, s):
    lines = [f"📊 Backtest {symbol} {interval} | {start} → {end or 'aujourd’hui'} | données: {source}",
             f"Capital : {s['capital_start']:.0f} → {s['capital_end']:.0f} ({s['return_pct']:+.2f} %)",
             f"Buy & hold : {s['buy_hold_pct']:+.2f} %",
             f"Max drawdown : {s['max_drawdown_pct']:.2f} %"]
    if s["trades"]:
        pf = "∞" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        lines += [f"Trades : {s['trades']} (long {s['longs']} / short {s['shorts']}) | gagnants {s['win_rate_pct']:.1f} %",
                  f"Espérance : {s['avg_r']:+.2f} R par trade | profit factor {pf}",
                  f"Durée moyenne : {s['avg_bars']:.1f} bougies | frais payés : {s['fees_paid']:.0f}"]
    else:
        lines.append("Aucun trade pris sur la période.")
    return "\n".join(lines)


def plot_equity(equity, df, path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    base = equity.iloc[0]
    ax.plot(equity.index, equity / base * 100, label="Stratégie", linewidth=1.4)
    bh = df.set_index("time")["close"]
    ax.plot(bh.index, bh / bh.iloc[0] * 100, label="Buy & hold", linewidth=1, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Capital (base 100)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def telegram(text=None, photo=None):
    """Envoie un message / une image et AFFICHE la réponse de Telegram dans les logs (pour diagnostiquer)."""
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        print("TELEGRAM: BOT_TOKEN ou CHAT_ID manquant (secrets absents de ce dépôt ?) -> envoi ignoré.")
        return False
    ok = True
    try:
        if text:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": text[:4000]}, timeout=20)
            print(f"TELEGRAM sendMessage -> {r.status_code} {r.text[:200]}")
            ok = ok and r.status_code == 200
        if photo:
            with open(photo, "rb") as f:
                r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                                  data={"chat_id": chat}, files={"photo": f}, timeout=30)
            print(f"TELEGRAM sendPhoto -> {r.status_code} {r.text[:200]}")
            ok = ok and r.status_code == 200
    except requests.RequestException as e:
        print("TELEGRAM erreur réseau:", e)
        return False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--source", default="auto", choices=["auto", "binance_vision", "kucoin", "bybit"])
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--risk", type=float, default=1.0, help="risque par trade en %% du capital")
    ap.add_argument("--fee", type=float, default=0.1, help="frais par ordre en %%")
    ap.add_argument("--slippage", type=float, default=0.02, help="glissement en %%")
    ap.add_argument("--leverage", type=float, default=1.0, help="levier maximum")
    ap.add_argument("--csv", default="", help="CSV local (time,open,high,low,close,volume) au lieu du téléchargement")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--params", default="", help="réglages de la stratégie : 'trend_mode=ma;tp_fib=0.62;confirms=bubble'")
    a = ap.parse_args()

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    os.makedirs(a.outdir, exist_ok=True)
    overrides = apply_params(a.params)
    ov_txt = ("Réglages : " + "; ".join(f"{k}={v}" for k, v in overrides.items())) if overrides else "Réglages : par défaut"
    print(ov_txt)
    # message de test immédiat : confirme que le bot Telegram et le CHAT_ID fonctionnent
    telegram(f"🚀 Backtest lancé : {', '.join(symbols)} {a.interval} depuis {a.start}\n{ov_txt}")
    recap = []
    for sym in symbols:
        try:
            if a.csv:
                df, source = data.load_csv(a.csv), "csv"
            else:
                df, source = data.load(sym, a.interval, a.start, a.end or None, a.source)
            trades, equity, stats = engine.run_backtest(
                df, strategy, capital=a.capital, risk_pct=a.risk, fee_pct=a.fee, slippage_pct=a.slippage,
                max_leverage=a.leverage, **strategy.ENGINE)
        except Exception as e:
            msg = f"❌ {sym} {a.interval} : {e}"
            print(msg)
            telegram(msg)
            recap.append(f"{sym}: erreur")
            continue
        summary = fmt_summary(sym, a.interval, source, a.start, a.end, stats)
        summary += "\n" + ov_txt
        bd = setup_breakdown(trades)
        if bd:
            summary += "\n" + bd
        print(summary)
        tag = f"{sym}_{a.interval}"
        trades.to_csv(os.path.join(a.outdir, f"{tag}_trades.csv"), index=False)
        with open(os.path.join(a.outdir, f"{tag}_resume.txt"), "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        png = os.path.join(a.outdir, f"{tag}_equity.png")
        plot_equity(equity, df, png, f"{sym} {a.interval} — {stats['return_pct']:+.1f} % (buy & hold {stats['buy_hold_pct']:+.1f} %)")
        telegram(summary, png)
        recap.append(f"{sym}: {stats['return_pct']:+.1f} % | {stats['trades']} trades | DD {stats['max_drawdown_pct']:.1f} %")

    if len(symbols) > 1:
        text = f"📋 Récapitulatif {a.interval} \n" + "\n".join(recap)
        print(text)
        telegram(text)


if __name__ == "__main__":
    main()
