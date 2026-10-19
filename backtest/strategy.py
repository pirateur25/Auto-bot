# backtest/strategy.py
# >>> C'EST LE SEUL FICHIER À MODIFIER À CHAQUE SIMULATION <<<
#
# ATTENTION : la stratégie ci-dessous n'est qu'un EXEMPLE pour vérifier que tout fonctionne.
# Ce n'est pas ta stratégie ; on la remplacera quand tu me l'auras expliquée.
#
# Contrat avec le moteur :
#   prepare(df)  -> renvoie le DataFrame avec les colonnes d'indicateurs ajoutées (calcul vectorisé).
#   signal(d, i) -> appelée à la CLÔTURE de la bougie i. d = dictionnaire de tableaux numpy
#                   (d["close"][i], d["bb_up"][i], ...). Ne lire QUE les indices <= i (sinon : biais du futur).
#                   Renvoie None, ou un dictionnaire :
#                      {"side": "long" | "short", "sl": prix_du_stop, "tp": prix_cible}
#                   ou, à la place de "tp", {"rr": 2.0} pour viser 2 fois le risque.
#                   L'ordre sera exécuté à l'ouverture de la bougie i+1.

import numpy as np

# Paramètres modifiables (à ajuster à chaque simulation)
PARAMS = {
    "bb_len": 20,        # longueur des bandes de Bollinger
    "bb_mult": 2.0,      # nombre d'écarts-types
    "trend_len": 200,    # moyenne mobile de tendance
    "atr_len": 14,
    "sl_atr": 0.5,       # marge du stop sous/sur la mèche de la bougie signal, en ATR
    "rr": 2.0,           # objectif : 2 fois le risque
}

# Réglages du moteur (utilisés par run_backtest.py)
ENGINE = {
    "allow_short": False,
    "max_bars": None,    # sortie forcée après N bougies (None = pas de limite)
}


def prepare(df):
    p = PARAMS
    close = df["close"]
    mid = close.rolling(p["bb_len"]).mean()
    std = close.rolling(p["bb_len"]).std(ddof=0)  # écart-type de population, comme TradingView
    df["bb_mid"] = mid
    df["bb_up"] = mid + p["bb_mult"] * std
    df["bb_dn"] = mid - p["bb_mult"] * std
    df["sma_trend"] = close.rolling(p["trend_len"]).mean()
    prev_close = close.shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()))
    df["atr"] = tr.rolling(p["atr_len"]).mean()
    body_low = np.minimum(df["open"], df["close"])
    body_high = np.maximum(df["open"], df["close"])
    # bulle apesanteur : tout le CORPS hors de la bande (mèches ignorées)
    df["bubble_up"] = (body_low > df["bb_up"]).astype(float)
    df["bubble_dn"] = (body_high < df["bb_dn"]).astype(float)
    return df


def signal(d, i):
    p = PARAMS
    atr, sma = d["atr"][i], d["sma_trend"][i]
    if np.isnan(atr) or np.isnan(sma) or np.isnan(d["bb_up"][i]):
        return None
    # EXEMPLE : bulle basse dans une tendance haussière -> achat de rebond vers l'objectif
    if d["bubble_dn"][i] == 1 and d["close"][i] > sma:
        return {"side": "long", "sl": d["low"][i] - p["sl_atr"] * atr, "rr": p["rr"]}
    return None
  
