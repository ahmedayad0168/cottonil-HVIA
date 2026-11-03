"""Basket cross-sell: shrunk item->item confidence with noisy-or aggregation.

score(j | basket S) = 1 - prod_{i in S} (1 - conf[i, j])            (probability-like)
conf[i, j]          = count(i & j) / (count(i) + k_shrink)          (shrinkage tames rare items)
ranking             = score * margin_per_unit[j]                     (expected margin), optionally by score alone
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def basket_matrix(order_ids: np.ndarray, prod_idx: np.ndarray, n_products: int) -> sp.csr_matrix:
    codes, uniq = pd.factorize(order_ids)
    m = sp.csr_matrix((np.ones(len(codes), dtype=np.float32), (codes, prod_idx)), shape=(len(uniq), n_products))
    m.data[:] = 1.0                      # binary (an order contains a product or not)
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def fit(B: sp.csr_matrix, k_shrink: float = 10.0) -> dict:
    co = (B.T @ B).toarray().astype(np.float64)
    n = np.diag(co).copy()
    np.fill_diagonal(co, 0)
    conf = co / (n[:, None] + k_shrink)
    N = B.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where((n[:, None] > 0) & (n[None, :] > 0), co * N / (n[:, None] * n[None, :]), 0.0)
    return {"conf": conf.astype(np.float32), "lift": lift.astype(np.float32), "count": co.astype(np.float32),
            "n_i": n.astype(np.float32), "n_orders": int(N), "k_shrink": k_shrink}


def score_baskets(conf: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """seed: (n_baskets x P) binary. Returns noisy-or probabilities (n_baskets x P)."""
    L = np.log1p(-np.clip(conf, 0, 0.999))
    s = seed @ L
    return 1.0 - np.exp(s)


def recommend(bundle: dict, basket: list[str], k: int = 5, available: set[str] | None = None,
              by: str = "margin") -> pd.DataFrame:
    """basket: list of product_ids. available: product_ids allowed (in-stock mask)."""
    pid = bundle["product_ids"]
    idx = {p: i for i, p in enumerate(pid)}
    seed = np.zeros((1, len(pid)), dtype=np.float32)
    for p in basket:
        if p in idx:
            seed[0, idx[p]] = 1
    if seed.sum() == 0:
        return pd.DataFrame(columns=["product_id", "prob", "expected_margin"])
    p = score_baskets(bundle["conf"], seed)[0]
    mpu = bundle["margin_per_unit"]
    p[seed[0] > 0] = 0
    if available is not None:
        mask = np.array([x in available for x in pid])
        p = np.where(mask, p, 0)
    em = p * mpu
    order = np.argsort(-(em if by == "margin" else p))[:k]
    order = [i for i in order if p[i] > 0]
    return pd.DataFrame({"product_id": [pid[i] for i in order], "prob": p[order], "expected_margin": em[order],
                         "margin_per_unit": mpu[order]})
