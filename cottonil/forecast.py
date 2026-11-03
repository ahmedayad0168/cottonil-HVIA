"""Weekly demand forecasting: panel construction, features, intervals.

Design (simple, robust, explainable):
  * one row per (branch x category x week)  -> "series"
  * one gradient-boosting model per horizon h = 1..4 weeks ("direct" forecasting,
    no error accumulation from recursive predictions)
  * Poisson loss (units are non-negative counts)
  * P10/P90 from a negative-binomial built on the model mean + an over-dispersion
    parameter estimated on a validation window
The same functions are used by notebook 01, scripts/batch_forecast.py and the tests.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .calendar_features import weekly_calendar_features

HORIZONS = (1, 2, 3, 4)
MIN_HISTORY_WEEKS = 26
CAT_FEATURES = ["branch_code", "category_code"]


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------
def make_panel(lines: pd.DataFrame, branches: pd.DataFrame | None = None):
    """Weekly units tensor.

    lines : needs columns order_date, branch_id, category, quantity
    Returns dict with U (S x W units), active (S x W bool), weeks (DatetimeIndex),
    series (DataFrame: branch_id, category, branch_code, category_code, loc_code, gov_code).
    Partial first/last weeks are dropped so a half-week never looks like a demand drop.
    """
    d = pd.to_datetime(lines["order_date"])
    first_day, last_day = d.min().normalize(), d.max().normalize()
    wk = (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.normalize()
    w0 = wk.min() + (pd.Timedelta(days=7) if first_day.weekday() != 0 else pd.Timedelta(0))
    w1 = wk.max() - (pd.Timedelta(days=7) if last_day.weekday() != 6 else pd.Timedelta(0))
    weeks = pd.date_range(w0, w1, freq="7D")
    W = len(weeks)

    branch_ids = sorted(lines["branch_id"].astype(str).unique())
    cats = sorted(lines["category"].astype(str).unique())
    B, K = len(branch_ids), len(cats)
    b_code = pd.Categorical(lines["branch_id"].astype(str), categories=branch_ids).codes.astype(np.int64)
    c_code = pd.Categorical(lines["category"].astype(str), categories=cats).codes.astype(np.int64)
    w_code = ((wk - w0).dt.days // 7).values.astype(np.int64)
    ok = (w_code >= 0) & (w_code < W)
    U = np.zeros((B * K, W), dtype=np.float32)
    np.add.at(U, (b_code[ok] * K + c_code[ok], w_code[ok]), lines["quantity"].values[ok].astype(np.float32))

    branch_tot = U.reshape(B, K, W).sum(1)
    first_active = np.where(branch_tot.sum(1) > 0, (branch_tot > 0).argmax(1), W)
    active = np.arange(W)[None, :] >= first_active[:, None]
    active = np.repeat(active, K, axis=0)

    series = pd.DataFrame({"branch_id": np.repeat(branch_ids, K), "category": np.tile(cats, B),
                           "branch_code": np.repeat(np.arange(B), K), "category_code": np.tile(np.arange(K), B)})
    if branches is not None:
        bm = branches.set_index("branch_id")
        loc = pd.Categorical(bm.loc[branch_ids, "retail_location_type"].astype(str)).codes.astype(np.int64)
        gov = pd.Categorical(bm.loc[branch_ids, "governorate_ar"].astype(str)).codes
        series["loc_code"] = np.repeat(loc, K)
        series["gov_code"] = np.repeat(gov, K)
    else:
        series["loc_code"] = 0
        series["gov_code"] = 0
    return {"U": U, "active": active, "weeks": weeks, "series": series, "B": B, "K": K,
            "branch_ids": branch_ids, "categories": cats}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _roll_mean(C, n):
    out = np.full((C.shape[0], C.shape[1] - 1), np.nan, dtype=np.float32)
    out[:, n - 1:] = (C[:, n:] - C[:, :-n]) / n
    return out


def _lag(U, k):
    out = np.full_like(U, np.nan)
    if k == 0:
        return U.copy()
    out[:, k:] = U[:, :-k]
    return out


def feature_names(h: int | None = None) -> list[str]:
    return ["branch_code", "category_code", "loc_code", "gov_code", "series_age_w",
            "lag0", "lag1", "lag2", "lag3", "lag7", "lag12",
            "m4", "m8", "m13", "m26", "sd8", "nz13", "trend_4_13", "branch_m4", "cat_net_m4", "share_of_branch",
            "ly_target", "ly_target_3wk",
            "t_month", "t_week", "t_ramadan", "t_days_eid_fitr", "t_days_eid_adha", "t_days_black_friday"]


def build_features(panel: dict, h: int, w_idx: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features for forecast origins `w_idx` (indices into panel['weeks']) and horizon h.

    Returns (X, y, keep): y is NaN where the target week lies beyond the data;
    `keep` is a boolean mask of the rows that are active series with enough history
    (X is already filtered with it; use `keep` to align meta columns).
    """
    U, A, weeks, series, B, K = panel["U"], panel["active"], panel["weeks"], panel["series"], panel["B"], panel["K"]
    S, W = U.shape
    w_idx = np.asarray(w_idx)
    C = np.concatenate([np.zeros((S, 1), np.float32), np.cumsum(U, axis=1, dtype=np.float64).astype(np.float32)], axis=1)
    C2 = np.concatenate([np.zeros((S, 1), np.float32), np.cumsum(U.astype(np.float64) ** 2, axis=1).astype(np.float32)], axis=1)
    m4, m8, m13, m26 = (_roll_mean(C, n) for n in (4, 8, 13, 26))
    sd8 = np.sqrt(np.maximum(_roll_mean(C2, 8) - m8 ** 2, 0))
    nz = (U > 0).astype(np.float32)
    Cn = np.concatenate([np.zeros((S, 1), np.float32), np.cumsum(nz, axis=1, dtype=np.float32)], axis=1)
    nz13 = _roll_mean(Cn, 13)
    branch_tot = U.reshape(B, K, W).sum(1)
    cb = np.concatenate([np.zeros((B, 1), np.float32), np.cumsum(branch_tot, axis=1, dtype=np.float32)], axis=1)
    branch_m4 = np.repeat(_roll_mean(cb, 4), K, axis=0)
    cat_tot = U.reshape(B, K, W).sum(0)
    cc = np.concatenate([np.zeros((K, 1), np.float32), np.cumsum(cat_tot, axis=1, dtype=np.float32)], axis=1)
    cat_m4 = np.tile(_roll_mean(cc, 4), (B, 1))
    first_active = np.where(A.any(1), A.argmax(1), W)

    def take(M):                      # (S, W) -> (S, len(w_idx)) at the requested origins
        return M[:, w_idx]

    def ly(k):                        # units one year before the *target* week (+k weeks)
        idx = w_idx + h - 52 + k
        out = np.full((S, len(w_idx)), np.nan, dtype=np.float32)
        ok = (idx >= 0) & (idx <= w_idx)          # must already be observed at origin time
        out[:, ok] = U[:, idx[ok]]
        return out

    ly0 = ly(0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ly3 = np.nanmean(np.stack([ly(-1), ly0, ly(1)]), axis=0)
    # calendar of the TARGET week (known in advance)
    last = weeks[-1]
    tgt_weeks = pd.DatetimeIndex([weeks[0] + pd.Timedelta(days=7 * int(i + h)) for i in w_idx])
    cal = weekly_calendar_features(tgt_weeks)
    n = len(w_idx)
    cols = {
        "branch_code": np.repeat(series["branch_code"].values[:, None], n, 1),
        "category_code": np.repeat(series["category_code"].values[:, None], n, 1),
        "loc_code": np.repeat(series["loc_code"].values[:, None], n, 1),
        "gov_code": np.repeat(series["gov_code"].values[:, None], n, 1),
        "series_age_w": w_idx[None, :] - first_active[:, None],
        "lag0": take(_lag(U, 0)), "lag1": take(_lag(U, 1)), "lag2": take(_lag(U, 2)), "lag3": take(_lag(U, 3)),
        "lag7": take(_lag(U, 7)), "lag12": take(_lag(U, 12)),
        "m4": take(m4), "m8": take(m8), "m13": take(m13), "m26": take(m26), "sd8": take(sd8), "nz13": take(nz13),
        "trend_4_13": take(m4) / (take(m13) + .1), "branch_m4": take(branch_m4), "cat_net_m4": take(cat_m4),
        "share_of_branch": take(m4) / (take(branch_m4) + .1),
        "ly_target": ly0, "ly_target_3wk": ly3,
        "t_month": np.repeat(cal["month"].values[None, :], S, 0), "t_week": np.repeat(cal["week_of_year"].values[None, :], S, 0),
        "t_ramadan": np.repeat(cal["is_ramadan"].values[None, :], S, 0),
        "t_days_eid_fitr": np.repeat(cal["days_to_eid_fitr"].values[None, :], S, 0),
        "t_days_eid_adha": np.repeat(cal["days_to_eid_adha"].values[None, :], S, 0),
        "t_days_black_friday": np.repeat(cal["days_to_black_friday"].values[None, :], S, 0),
    }
    X = pd.DataFrame({k: v.ravel().astype(np.float32) for k, v in cols.items()})[feature_names()]
    tgt = w_idx + h
    y = np.full((S, n), np.nan, dtype=np.float32)
    inside = tgt < W
    y[:, inside] = U[:, tgt[inside]]
    y = y.ravel()
    act = A[:, w_idx].ravel()
    enough = (np.repeat(w_idx[None, :], S, 0).ravel() >= MIN_HISTORY_WEEKS - 1)
    keep = act & enough
    meta = pd.DataFrame({"s": np.repeat(np.arange(S), n), "w": np.tile(w_idx, S)})
    return X[keep].reset_index(drop=True), y[keep], meta[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# metrics + intervals
# ---------------------------------------------------------------------------
def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))


def bias(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(p.sum() / max(y.sum(), 1e-9) - 1)


def estimate_overdispersion(y, mu) -> float:
    """Method-of-moments phi for Var(y|mu) = mu + phi*mu^2 (phi >= 0)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float(max(0.0, (((y - mu) ** 2 - mu).sum()) / max((mu ** 2).sum(), 1e-9)))


def nb_quantile(mu, phi: float, q: float) -> np.ndarray:
    mu = np.maximum(np.asarray(mu, float), 1e-6)
    if phi < 1e-6:
        return stats.poisson.ppf(q, mu)
    n = 1.0 / phi
    return stats.nbinom.ppf(q, n, n / (n + mu))


def predict_horizon(bundle: dict, panel: dict, h: int, w_idx: np.ndarray) -> pd.DataFrame:
    """Mean / P10 / P90 for horizon h from a saved bundle."""
    X, y, meta = build_features(panel, h, w_idx)
    mu = bundle["models"][h].predict(X[bundle["features"]])
    phi = bundle["phi"][h]
    out = meta.copy()
    out["mean"] = mu
    # discrete quantiles of tiny-volume series can fall on the wrong side of the mean (e.g. P90 = 0, mean = 0.2);
    # clamp so the band always brackets the forecast
    out["p10"] = np.minimum(nb_quantile(mu, phi, .10), mu)
    out["p90"] = np.maximum(nb_quantile(mu, phi, .90), mu)
    out["actual"] = y
    return out


def latest_forecast(bundle: dict, panel: dict) -> pd.DataFrame:
    """Forecast weeks W..W+3 for every active series, from the last observed week."""
    W = panel["U"].shape[1]
    rows = []
    for h in HORIZONS:
        o = predict_horizon(bundle, panel, h, np.array([W - 1]))
        o["horizon"] = h
        o["week_start"] = panel["weeks"][-1] + pd.Timedelta(days=7 * h)
        rows.append(o)
    f = pd.concat(rows, ignore_index=True)
    s = panel["series"]
    f["branch_id"] = s["branch_id"].values[f["s"].values]
    f["category"] = s["category"].values[f["s"].values]
    return f[["branch_id", "category", "horizon", "week_start", "mean", "p10", "p90"]]
