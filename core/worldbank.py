"""
Optional live World Bank pull — proves API integration for reports.

Uses total trade (% of GDP) as a lagged macro sanity check (not bilateral).
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests

WB_BASE = "https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}"


def fetch_trade_percent_gdp(iso2: str, years: int = 3) -> List[Dict[str, Any]]:
    """
    NE.TRD.GNFS.ZS — Trade (% of GDP). iso2 e.g. IN, US (World Bank uses 2-letter for many).
    Returns newest `years` annual observations (may be empty on network error).
    """
    iso = iso2.upper()
    url = WB_BASE.format(iso=iso, indicator="NE.TRD.GNFS.ZS")
    params = {"format": "json", "per_page": 20, "mrnev": years}
    rows: List[Dict[str, Any]] = []
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return rows
        for item in data[1] or []:
            if not item:
                continue
            rows.append(
                {
                    "iso2": iso,
                    "year": item.get("date"),
                    "trade_pct_gdp": item.get("value"),
                }
            )
    except requests.RequestException:
        pass
    return rows


def macro_context_line(iso2: str) -> str:
    pts = fetch_trade_percent_gdp(iso2, years=2)
    if not pts or pts[0].get("trade_pct_gdp") is None:
        return f"{iso2}: (World Bank macro fetch unavailable)"
    latest = pts[0]
    return (
        f"{iso2}: Trade was {latest['trade_pct_gdp']:.1f}% of GDP in {latest['year']} "
        f"(World Bank NE.TRD.GNFS.ZS; bilateral shares still come from local snapshot)."
    )
