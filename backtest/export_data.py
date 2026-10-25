# backtest/export_data.py
# Télécharge l'historique de bougies (même source que le backtest), le range dans un zip par paire,
# et envoie ce zip sur Telegram (limite 50 Mo) pour que tu puisses le récupérer sur ton téléphone.
#
# Exemple : python backtest/export_data.py --symbols BTCUSDT --interval 5m --start 2021-01-01

import argparse
import os
import sys
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402

TELEGRAM_LIMIT = 49 * 1024 * 1024


def telegram_document(path, caption):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        print("TELEGRAM: BOT_TOKEN ou CHAT_ID manquant -> envoi ignoré (le fichier reste dans les artefacts GitHub).")
        return False
    with open(path, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendDocument", data={"chat_id": chat, "caption": caption},
                          files={"document": f}, timeout=300)
    print(f"TELEGRAM sendDocument -> {r.status_code} {r.text[:200]}")
    return r.status_code == 200


def export(symbol, interval, start, end, outdir, source="auto"):
    df, src = data.load(symbol, interval, start, end or None, source, use_cache=False)
    os.makedirs(outdir, exist_ok=True)
    name = f"{symbol}_{interval}_{df['time'].iloc[0]:%Y-%m-%d}_{df['time'].iloc[-1]:%Y-%m-%d}"
    csv_path = os.path.join(outdir, name + ".csv")
    out = df.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out.to_csv(csv_path, index=False)
    zip_path = os.path.join(outdir, name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(csv_path, os.path.basename(csv_path))
    os.remove(csv_path)
    size = os.path.getsize(zip_path)
    print(f"{symbol}: {len(df)} bougies {interval}, {df['time'].iloc[0]} -> {df['time'].iloc[-1]}, source {src}, zip {size / 1e6:.1f} Mo")
    return zip_path, len(df), src, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--source", default="auto")
    ap.add_argument("--outdir", default="data_out")
    a = ap.parse_args()
    for sym in [s.strip().upper() for s in a.symbols.split(",") if s.strip()]:
        try:
            path, n, src, size = export(sym, a.interval, a.start, a.end, a.outdir, a.source)
        except Exception as e:
            print(f"❌ {sym}: {e}")
            continue
        cap = f"📦 Données {sym} {a.interval} : {n} bougies ({src}). Joins ce fichier à ta conversation avec Claude."
        if size > TELEGRAM_LIMIT:
            print(f"{sym}: zip trop gros pour Telegram ({size / 1e6:.0f} Mo > 49 Mo) : prends-le dans les artefacts GitHub.")
        else:
            telegram_document(path, cap)


if __name__ == "__main__":
    main()
