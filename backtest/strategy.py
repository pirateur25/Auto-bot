# backtest/strategy.py
# Stratégie : tendance H4 + retracement (zone OTE) + confirmation (bulle apesanteur ou MSS) sur 5m / 15m / 1h.
#
# Raisonnement écrit pour une tendance BAISSIÈRE (position short). Le cas haussier est obtenu en
# miroir (prix inversés) : exactement la même logique, donc aucune différence de traitement.
#
# Principe, dans l'ordre :
#  1) TENDANCE H4 baissière (loi de Dow = creux et sommets de plus en plus bas, et/ou moyenne mobile).
#  2) RETRACEMENT : on repère la dernière jambe baissière (dernier sommet confirmé -> plus bas),
#     et on attend que le prix remonte dans la zone OTE (61,8 % - 78,6 % de cette jambe).
#     Les jambes sont calculées sur H4 et sur H1 (plus de trades).
#  3) CONFIRMATION dans la zone, sur 5m, 15m ou 1h :
#       - "bubble" : bulle apesanteur HAUTE (corps entièrement au-dessus de la bande de Bollinger haute)
#                    suivie d'une bougie BAISSIÈRE  -> entrée à la clôture de cette bougie ;
#       - "mss"    : le prix balaie un sommet (sweep) puis clôture sous le creux précédent (cassure de
#                    structure), et cette cassure est plus rapide que la montée qui a fait le sweep.
#  4) SL au plus haut local du retracement (+ petite marge), TP dans la zone OTE du mouvement de
#     retracement (par défaut au niveau 70,5 % de ce mouvement).
#
# Contrat avec le moteur : prepare(df) calcule tout (sans regarder le futur), signal(d, i) lit le résultat.
# Les paramètres se modifient ici, ou depuis le workflow GitHub (champ « params »), ex :
#   trend_mode=ma;tp_fib=0.62;confirms=bubble

import numpy as np
import pandas as pd

PARAMS = {
    # --- tendance ---
    "trend_tf": "4h",
    "trend_mode": "dow",        # "dow" (structure) | "ma" (moyenne mobile) | "both" (les deux d'accord)
    "ma_len": 50,               # EMA de tendance (mode ma / both)
    "ma_slope_bars": 5,         # la pente de l'EMA est mesurée sur N bougies H4
    # --- swings (ZigZag) : un retournement est confirmé quand le prix repart de N x ATR ---
    "atr_len": 14,
    "zz_atr_4h": 2.0,
    "zz_atr_1h": 2.5,
    "zz_atr_15m": 3.0,
    # --- retracement ---
    "range_tfs": "4h,1h",       # jambes utilisées pour la zone OTE
    "ote_lo": 0.618,
    "ote_hi": 0.786,
    # --- confirmations ---
    "confirms": "bubble,mss",   # "bubble", "mss" ou les deux
    "conf_tfs": "5m,15m,1h",    # unités de temps de confirmation (>= unité de base du backtest)
    "bb_len": 20,
    "bb_mult": 2.0,
    # bulle validée = corps hors bande ET distance >= seuil (en % du prix). Seuils tirés de ton tableau :
    "bubble_min_pct_5m": 0.045,
    "bubble_min_pct_15m": 0.01,
    "bubble_min_pct_1h": 0.05,
    "bubble_min_sigma": 0.0,    # seuil optionnel en écarts-types (0 = désactivé)
    "mss_fractal": 2,           # un creux/sommet est confirmé après N bougies de chaque côté
    "mss_speed_ratio": 1.0,     # cassure en <= ratio x (durée de la montée du sweep)
    # --- sortie ---
    "sl_buf_atr": 0.2,          # marge au-dessus du plus haut local, en ATR de l'unité de confirmation
    "tp_fib": 0.705,            # TP = retracement de ce niveau du mouvement de retracement (0.618 à 0.786)
    "min_rr": 0.0,              # ratio gain/risque minimum (0 = pas de filtre)
    "side": "both",             # "both" | "long" | "short"
}

ENGINE = {"allow_short": True, "max_bars": None}

TF_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
NS_MIN = 60 * 10**9


def _csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


# ------------------------------------------------------------------ données multi-unités de temps

def make_frame(df, minutes, base_min):
    """Bougies de l'unité demandée, uniquement les bougies COMPLÈTES, avec leurs heures d'ouverture / clôture."""
    g = df.set_index("time")
    if minutes == base_min:
        f = g[["open", "high", "low", "close"]].copy()
    else:
        f = g.resample(f"{minutes}min").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                           close=("close", "last"), n=("close", "count"))
        f = f[f["n"] == minutes // base_min][["open", "high", "low", "close"]]
    open_ns = pd.DatetimeIndex(f.index).as_unit("ns").asi8
    return {"o": f["open"].to_numpy(float), "h": f["high"].to_numpy(float), "l": f["low"].to_numpy(float),
            "c": f["close"].to_numpy(float), "open_ns": open_ns, "close_ns": open_ns + minutes * NS_MIN}


def add_indicators(f, p, tf):
    o, h, l, c = f["o"], f["h"], f["l"], f["c"]
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    f["atr"] = pd.Series(tr).rolling(p["atr_len"]).mean().to_numpy()
    s = pd.Series(c)
    mid = s.rolling(p["bb_len"]).mean()
    sd = s.rolling(p["bb_len"]).std(ddof=0).to_numpy()  # écart-type de population, comme TradingView
    up, dn = (mid + p["bb_mult"] * sd).to_numpy(), (mid - p["bb_mult"] * sd).to_numpy()
    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    thr = p.get(f"bubble_min_pct_{tf}", 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_up, d_dn = body_lo - up, dn - body_hi
        f["bub_up"] = (d_up > 0) & (d_up / body_lo * 100 >= thr) & (d_up / sd >= p["bubble_min_sigma"])
        f["bub_dn"] = (d_dn > 0) & (d_dn / body_hi * 100 >= thr) & (d_dn / sd >= p["bubble_min_sigma"])


# ------------------------------------------------------------------ swings (ZigZag causal)

def zigzag(h, l, atr, mult):
    """Sommets / creux confirmés quand le prix repart de mult x ATR. Tout est causal : l'état à la bougie t
    ne dépend que des bougies <= t. Renvoie, pour chaque bougie, la direction de la jambe en cours (d = +1
    montante, -1 descendante), son extrême provisoire (ext) et les 2 derniers sommets / creux confirmés."""
    n = len(h)
    nan = np.nan
    out = {k: np.full(n, nan) for k in ("ext", "cur_hi", "prev_hi", "cur_lo", "prev_lo")}
    out.update({k: np.full(n, -1, dtype=np.int64) for k in ("ext_i", "cur_hi_i", "cur_lo_i")})
    out["d"] = np.zeros(n, dtype=np.int8)
    d = 0
    started = False
    hh = ll = ext = nan
    hh_i = ll_i = ext_i = -1
    cur_hi = (nan, -1)
    prev_hi = (nan, -1)
    cur_lo = (nan, -1)
    prev_lo = (nan, -1)
    for t in range(n):
        a = atr[t]
        if a != a:
            continue
        thr = mult * a
        if not started:
            hh, hh_i, ll, ll_i, started = h[t], t, l[t], t, True
        elif d == 0:
            if h[t] > hh:
                hh, hh_i = h[t], t
            if l[t] < ll:
                ll, ll_i = l[t], t
            if hh - ll >= thr:
                if ll_i < hh_i:
                    cur_lo, d, ext, ext_i = (ll, ll_i), 1, hh, hh_i
                elif hh_i < ll_i:
                    cur_hi, d, ext, ext_i = (hh, hh_i), -1, ll, ll_i
        elif d == 1:
            if h[t] > ext:
                ext, ext_i = h[t], t
            if ext - l[t] >= thr:
                prev_hi, cur_hi = cur_hi, (ext, ext_i)
                d, ext, ext_i = -1, l[t], t
        else:
            if l[t] < ext:
                ext, ext_i = l[t], t
            if h[t] - ext >= thr:
                prev_lo, cur_lo = cur_lo, (ext, ext_i)
                d, ext, ext_i = 1, h[t], t
        out["d"][t] = d
        out["ext"][t], out["ext_i"][t] = ext, ext_i
        out["cur_hi"][t], out["cur_hi_i"][t], out["prev_hi"][t] = cur_hi[0], cur_hi[1], prev_hi[0]
        out["cur_lo"][t], out["cur_lo_i"][t], out["prev_lo"][t] = cur_lo[0], cur_lo[1], prev_lo[0]
    return out


def trend_bear(T, th, tl, tc, p):
    """Tendance baissière à chaque bougie de l'unité de tendance (bool). Loi de Dow, moyenne mobile, ou les deux."""
    mode = p["trend_mode"]
    n = len(tc)
    bear_dow = np.zeros(n, bool)
    bear_ma = np.zeros(n, bool)
    if mode in ("dow", "both"):
        zz = zigzag(th, tl, T["atr"], p.get("zz_atr_" + p["trend_tf"], 2.0))
        have = ~(np.isnan(zz["prev_hi"]) | np.isnan(zz["prev_lo"]) | np.isnan(zz["cur_hi"]) | np.isnan(zz["cur_lo"]))
        with np.errstate(invalid="ignore"):
            # sommets ET creux de plus en plus bas ; tendance annulée si le prix clôture au-dessus du dernier sommet
            bear_dow = have & (zz["cur_hi"] < zz["prev_hi"]) & (zz["cur_lo"] < zz["prev_lo"]) & (tc <= zz["cur_hi"])
    if mode in ("ma", "both"):
        ema = pd.Series(tc).ewm(span=p["ma_len"], adjust=False).mean().to_numpy()
        k = p["ma_slope_bars"]
        slope = ema - np.concatenate([np.full(k, np.nan), ema[:-k]])
        with np.errstate(invalid="ignore"):
            bear_ma = (tc < ema) & (slope < 0)
    if mode == "dow":
        return bear_dow
    if mode == "ma":
        return bear_ma
    return bear_dow & bear_ma


def bear_range(F, fh, fl, p, tf):
    """Dernière jambe baissière : H = dernier sommet confirmé, L = plus bas depuis ce sommet.
    Valide tant que le prix n'a pas repris le sommet H."""
    zz = zigzag(fh, fl, F["atr"], p.get("zz_atr_" + tf, 2.0))
    d, H = zz["d"], zz["cur_hi"]
    n = len(d)
    L = np.full(n, np.nan)
    Li = np.full(n, -1, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        down = (d == -1) & ~np.isnan(H)
        up = ((d == 1) & ~np.isnan(H) & ~np.isnan(zz["cur_lo"]) & (zz["cur_lo_i"] > zz["cur_hi_i"]) & (zz["ext"] < H))
        L[down], Li[down] = zz["ext"][down], zz["ext_i"][down]
        L[up], Li[up] = zz["cur_lo"][up], zz["cur_lo_i"][up]
        valid = (down | up) & (L < H)
    return {"H": H, "L": L, "Li": Li, "valid": valid}


# ------------------------------------------------------------------ MSS après sweep

def mss_events(h, l, c, k, ratio):
    """MSS baissier : (1) un creux L2 confirmé, (2) le prix a balayé le sommet H1 précédent (sweep),
    (3) une bougie CLÔTURE sous L2 (première fois), (4) cette cassure est plus rapide que la montée qui a
    fait le sweep : (t - bougie du sweep) <= ratio x (bougie du sweep - L2). Causal : un pivot n'est connu
    qu'k bougies après sa formation. Renvoie (événement, prix du sweep)."""
    n = len(h)
    ev = np.zeros(n, bool)
    h2 = np.full(n, np.nan)
    highs = []            # sommets confirmés (indice, prix)
    low_i, low_p = -1, np.nan
    runmax, runmax_i = -np.inf, -1
    fired = -1
    for t in range(2 * k, n):
        p = t - k
        if l[p] < l[p - k:p].min() and l[p] <= l[p + 1:t + 1].min():
            low_i, low_p = p, l[p]
            seg = h[p + 1:t + 1]
            runmax, runmax_i = seg.max(), p + 1 + int(seg.argmax())
        else:
            if low_i >= 0 and h[t] > runmax:
                runmax, runmax_i = h[t], t
        if h[p] > h[p - k:p].max() and h[p] >= h[p + 1:t + 1].max():
            highs.append((p, h[p]))
        if low_i >= 0 and low_i != fired and c[t] < low_p <= c[t - 1]:
            h1 = None
            for hi, hp in reversed(highs):
                if hi < low_i:
                    h1 = hp
                    break
            if h1 is not None and runmax > h1 and (t - runmax_i) <= ratio * (runmax_i - low_i):
                ev[t], h2[t], fired = True, runmax, low_i
    return ev, h2


# ------------------------------------------------------------------ construction des signaux

def run_world(F, B, jm, p, sign):
    """sign=+1 : on cherche des SHORTS (tendance baissière). sign=-1 : monde en miroir -> des LONGS.
    Renvoie une liste de (indice_base, sl, tp, code) avec des prix déjà remis à l'échelle réelle."""
    def view(f):
        if sign == 1:
            return f["o"], f["h"], f["l"], f["c"]
        return -f["o"], -f["l"], -f["h"], -f["c"]

    confirms = _csv(p["confirms"])
    nb = len(B["c"])
    tT = p["trend_tf"]
    if tT not in F:
        return [], 0.0
    T = F[tT]
    _, th, tl, tc = view(T)
    bear = trend_bear(T, th, tl, tc, p)
    ranges = {}
    for R in _csv(p["range_tfs"]):
        if R in F:
            _, rh, rl, _ = view(F[R])
            ranges[R] = bear_range(F[R], rh, rl, p, R)
    _, bh, bl, _ = view(B)
    out = []
    for Y in _csv(p["conf_tfs"]):
        if Y not in F:
            continue
        V = F[Y]
        vo, vh, vl, vc = view(V)
        iY = np.searchsorted(B["close_ns"], V["close_ns"])
        okY = np.zeros(len(vo), bool)
        inside = iY < nb
        okY[inside] = B["close_ns"][iY[inside]] == V["close_ns"][inside]
        cands = []
        if "bubble" in confirms and len(vo) > 1:
            bub = V["bub_up"] if sign == 1 else V["bub_dn"]
            idx = np.where(bub[:-1] & (vc[1:] < vo[1:]))[0] + 1  # bulle puis bougie baissière
            cands += [(int(j), "bubble", None) for j in idx if okY[j]]
        if "mss" in confirms:
            ev, h2 = mss_events(vh, vl, vc, p["mss_fractal"], p["mss_speed_ratio"])
            cands += [(int(j), "mss", h2[j]) for j in np.where(ev)[0] if okY[j]]
        cands.sort(key=lambda x: x[0])
        for j, kind, sweep in cands:
            i = int(iY[j])
            jt = jm[tT][i]
            if jt < 0 or not bear[jt]:
                continue
            for R, rg in ranges.items():
                jr = jm[R][i]
                if jr < 0 or not rg["valid"][jr]:
                    continue
                H, L = rg["H"][jr], rg["L"][jr]
                zl, zh = L + p["ote_lo"] * (H - L), L + p["ote_hi"] * (H - L)
                if kind == "bubble":
                    if not (max(vh[j - 1], vh[j]) >= zl and min(vl[j - 1], vl[j]) <= zh):
                        continue
                elif not (zl <= sweep <= H):
                    continue
                Rf, li = F[R], int(rg["Li"][jr])
                bs = int(np.searchsorted(B["open_ns"], Rf["open_ns"][li], side="left"))
                be = int(np.searchsorted(B["open_ns"], Rf["close_ns"][li], side="left"))
                if be <= bs:
                    continue
                bL = bs + int(np.argmin(bl[bs:be]))
                after_l, after_h = bl[bL + 1:i + 1], bh[bL + 1:i + 1]
                if len(after_h) == 0 or after_l.min() < bl[bL]:  # nouveau plus bas : la jambe a changé
                    continue
                peak = after_h.max()
                if peak >= H:
                    continue
                atr = V["atr"][j]
                if atr != atr:
                    continue
                sl = peak + p["sl_buf_atr"] * atr
                tp = peak - p["tp_fib"] * (peak - bl[bL])
                entry = vc[j]
                if not (tp < entry < sl):
                    continue
                if p["min_rr"] and (entry - tp) < p["min_rr"] * (sl - entry):
                    continue
                out.append((i, sign * sl, sign * tp, f"{kind}{Y}|{R}"))
                break
    return out, float(bear.mean())


def prepare(df):
    p = PARAMS
    open_ns = pd.DatetimeIndex(df["time"]).as_unit("ns").asi8
    base_min = int(round(np.median(np.diff(open_ns)) / NS_MIN))
    B = {"o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float), "l": df["low"].to_numpy(float),
         "c": df["close"].to_numpy(float), "open_ns": open_ns, "close_ns": open_ns + base_min * NS_MIN}
    nb = len(df)
    dropped = [tf for tf in _csv(p["conf_tfs"]) if TF_MIN[tf] < base_min]
    if dropped:
        print(f"Unité de base {base_min} min : confirmations {', '.join(dropped)} ignorées (trop fines).")
    need = {p["trend_tf"], *_csv(p["range_tfs"]), *_csv(p["conf_tfs"])}
    F = {}
    for tf in need:
        if TF_MIN[tf] >= base_min:
            F[tf] = make_frame(df, TF_MIN[tf], base_min)
            add_indicators(F[tf], p, tf)
    jm = {tf: np.searchsorted(f["close_ns"], B["close_ns"], side="right") - 1 for tf, f in F.items()}

    setups = []
    for sign, side, name in ((1, -1, "short"), (-1, 1, "long")):
        if p["side"] not in ("both", name):
            continue
        res, share = run_world(F, B, jm, p, sign)
        setups += [(i, side, sl, tp, code) for i, sl, tp, code in res]
        label = "baissière" if sign == 1 else "haussière"
        print(f"Tendance {p['trend_tf']} {label} ({p['trend_mode']}) : {share * 100:.1f} % du temps | signaux {name} : {len(res)}")

    sig_side = np.zeros(nb, dtype=np.int8)
    sig_sl = np.full(nb, np.nan)
    sig_tp = np.full(nb, np.nan)
    sig_code = np.array([""] * nb, dtype=object)
    for i, side, sl, tp, code in sorted(setups, key=lambda x: x[0]):
        if sig_side[i] == 0:
            sig_side[i], sig_sl[i], sig_tp[i], sig_code[i] = side, sl, tp, code
    df["sig_side"], df["sig_sl"], df["sig_tp"], df["sig_code"] = sig_side, sig_sl, sig_tp, sig_code
    return df


def signal(d, i):
    s = d["sig_side"][i]
    if s == 0:
        return None
    return {"side": "long" if s > 0 else "short", "sl": d["sig_sl"][i], "tp": d["sig_tp"][i], "meta": d["sig_code"][i]}
