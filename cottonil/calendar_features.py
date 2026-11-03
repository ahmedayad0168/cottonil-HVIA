"""Egyptian retail calendar: Ramadan, Eid al-Fitr, Eid al-Adha, Black Friday.

Hijri dates are fixed tables (approximate +/- 1 day, which is irrelevant at
weekly granularity). Extend the dicts if you need years past 2028.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# year -> (first day of Ramadan, last day of Ramadan)
RAMADAN = {
    2018: ("2018-05-16", "2018-06-14"), 2019: ("2019-05-06", "2019-06-03"),
    2020: ("2020-04-24", "2020-05-23"), 2021: ("2021-04-13", "2021-05-12"),
    2022: ("2022-04-02", "2022-05-01"), 2023: ("2023-03-23", "2023-04-20"),
    2024: ("2024-03-11", "2024-04-09"), 2025: ("2025-03-01", "2025-03-29"),
    2026: ("2026-02-18", "2026-03-19"), 2027: ("2027-02-08", "2027-03-09"),
    2028: ("2028-01-28", "2028-02-26"), 2029: ("2029-01-16", "2029-02-14"),
}
EID_ADHA = {
    2018: "2018-08-21", 2019: "2019-08-11", 2020: "2020-07-31", 2021: "2021-07-20",
    2022: "2022-07-09", 2023: "2023-06-28", 2024: "2024-06-16", 2025: "2025-06-06",
    2026: "2026-05-27", 2027: "2027-05-16", 2028: "2028-05-05", 2029: "2029-04-24",
}


def black_friday(year: int) -> pd.Timestamp:
    """Last Friday of November."""
    d = pd.Timestamp(year=year, month=11, day=30)
    while d.weekday() != 4:
        d -= pd.Timedelta(days=1)
    return d


RAMADAN_START = pd.to_datetime([v[0] for v in RAMADAN.values()]).values
RAMADAN_END = pd.to_datetime([v[1] for v in RAMADAN.values()]).values
EID_FITR_DATES = (pd.to_datetime([v[1] for v in RAMADAN.values()]) + pd.Timedelta(days=1)).values
EID_ADHA_DATES = pd.to_datetime(list(EID_ADHA.values())).values
BLACK_FRIDAYS = pd.to_datetime([black_friday(y) for y in range(2018, 2030)]).values


def _days_to_next(dates: np.ndarray, events: np.ndarray, cap: int = 60) -> np.ndarray:
    """Days from each date to the next event (>= 0). Capped, so 'far away' = cap."""
    events = np.sort(events)
    idx = np.searchsorted(events, dates, side="left")
    out = np.full(len(dates), cap, dtype=float)
    ok = idx < len(events)
    out[ok] = (events[idx[ok]] - dates[ok]) / np.timedelta64(1, "D")
    return np.clip(out, 0, cap)


def calendar_features(dates) -> pd.DataFrame:
    """Per-date calendar features. `dates` = anything pd.to_datetime accepts."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    arr = d.values.astype("datetime64[ns]")
    in_ram = np.zeros(len(d), dtype=bool)
    for s, e in zip(RAMADAN_START, RAMADAN_END):
        in_ram |= (arr >= s) & (arr <= e)
    return pd.DataFrame(
        {
            "month": d.month.astype("int8"),
            "week_of_year": d.isocalendar().week.astype("int16").values,
            "is_ramadan": in_ram.astype("int8"),
            "days_to_eid_fitr": _days_to_next(arr, EID_FITR_DATES),
            "days_to_eid_adha": _days_to_next(arr, EID_ADHA_DATES),
            "days_to_black_friday": _days_to_next(arr, BLACK_FRIDAYS),
        },
        index=None,
    )


def weekly_calendar_features(week_start) -> pd.DataFrame:
    """Calendar features for weeks (Mon..Sun). Uses mid-week date + ramadan share."""
    ws = pd.DatetimeIndex(pd.to_datetime(week_start))
    mid = calendar_features(ws + pd.Timedelta(days=3))
    # share of the week's 7 days inside Ramadan (smoother than the mid-week flag)
    share = np.zeros(len(ws))
    for k in range(7):
        share += calendar_features(ws + pd.Timedelta(days=k))["is_ramadan"].values
    mid["is_ramadan"] = (share / 7.0).astype("float32")
    return mid
