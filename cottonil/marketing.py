"""Marketing response curves + budget optimizer.

Per channel c (campaign level):   log(orders) = a_c + beta_c * log(spend) + controls
beta < 1  => diminishing returns (doubling spend gives less than double the orders).
Annual channel response for total channel spend S spread evenly over n campaigns:
        orders_c(S) = n * exp(c0_c) * (S / n) ** beta_c
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def channel_orders(params: dict, channel: str, S: float) -> float:
    n, b, c0 = params["n"][channel], params["beta"][channel], params["c0"][channel]
    return float(n * np.exp(c0) * (max(S, 1e-9) / n) ** b)


def total_orders(params: dict, alloc: dict) -> float:
    return sum(channel_orders(params, c, s) for c, s in alloc.items())


def optimize(params: dict, total_budget: float | None = None, lo: float = 0.5, hi: float = 1.5, beta=None, c0=None) -> dict:
    """Maximise total orders for a fixed budget. Variables are *multipliers of current spend* (~1.0),
    which keeps SLSQP well scaled; each channel is limited to [lo, hi] x its current spend."""
    ch = list(params["channels"])
    cur = np.array([params["spend_current"][c] for c in ch], float)
    total = float(cur.sum() if total_budget is None else total_budget)
    p = dict(params)
    if beta is not None:
        p["beta"] = dict(zip(ch, beta))
    if c0 is not None:
        p["c0"] = dict(zip(ch, c0))
    base = sum(channel_orders(p, c, s) for c, s in zip(ch, cur))

    def f(u):                                    # negative relative orders
        return -sum(channel_orders(p, c, s * m) for c, s, m in zip(ch, cur, u)) / base

    u0 = np.clip(np.full(len(ch), total / cur.sum()), lo, hi)
    cons = ({"type": "eq", "fun": lambda u: (cur * u).sum() / cur.sum() - total / cur.sum()},)
    r = minimize(f, u0, method="SLSQP", bounds=[(lo, hi)] * len(ch), constraints=cons,
                 options={"maxiter": 500, "ftol": 1e-12})
    u = r.x if (r.success or f(r.x) < f(u0)) else u0
    alloc = dict(zip(ch, cur * u))
    return {"alloc": alloc, "orders": -f(u) * base, "orders_current": base, "success": bool(r.success)}
