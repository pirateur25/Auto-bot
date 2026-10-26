# backtest/convert_forex.py
# Convertit les CSV produits par dukascopy-cli (timestamp en ms, open, high, low, close, volume) au format du backtest
# (time, open, high, low, close, volume, heure UTC), les range dans un zip, et l'envoie sur Telegram.
#
# Exemple : python backtest/convert_forex.py --pair eurusd --interval 5m --src download --out data_out

import argparse
import glob
import os
import zipfile

import pandas as pd
import requests


def telegram_document(path, caption):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        print("TELEGRAM: BOT_TOKEN ou CHAT_ID manquant -> envoi ignoré (le fichier reste dans les artefacts GitHub).")
        return
    with open(path, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendDocument", data={"chat_id": chat, "caption": caption},
                          files={"document": f}, timeout=300)
    print(f"TELEGRAM sendDocument -> {r.status_code} {r.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--src", default="download")
    ap.add_argument("--out", default="data_out")
    a = ap.parse_args()
    a.interval = {"m5": "5m", "m15": "15m", "h1": "1h"}.get(a.interval, a.interval)
    files = sorted(glob.glob(os.path.join(a.src, "**", "*.csv"), recursive=True))
    files = [f for f in files if a.pair.lower() in os.path.basename(f).lower()] or files
    if not files:
        raise SystemExit("Aucun CSV trouvé : le téléchargement Dukascopy a échoué (voir les logs de l'étape précédente).")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = "timestamp" if "timestamp" in df.columns else df.columns[0]
    ts = pd.to_numeric(df[tcol], errors="coerce")
    if ts.notna().all():
        df["time"] = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df[tcol], utc=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.pair.upper()}_{a.interval}_{df['time'].iloc[0]:%Y-%m-%d}_{df['time'].iloc[-1]:%Y-%m-%d}"
    csv_path = os.path.join(a.out, name + ".csv")
    out = df.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out.to_csv(csv_path, index=False)
    zip_path = os.path.join(a.out, name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, os.path.basename(csv_path))
    print(f"{a.pair}: {len(df)} bougies {a.interval} du {df['time'].iloc[0]} au {df['time'].iloc[-1]} -> {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} Mo)")
    telegram_document(zip_path, f"{a.pair.upper()} {a.interval} ({len(df)} bougies)")


if __name__ == "__main__":
    main()
