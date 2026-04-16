"""
Target country pairs for bilateral corpus collection.

Countries: India (IN), China (CN), USA (US), Russia (RU),
           Pakistan (PK), Iran (IR), Israel (IL)

Pairs selected for strong English RSS coverage + geopolitical/economic relevance.
ISO2 codes; order in tuples is arbitrary (CSV stores sorted country_1, country_2).
"""

from __future__ import annotations

CORPUS_TARGET_PAIRS: list[tuple[str, str]] = [
    # Original core pairs
    ("IN", "CN"),   # India–China: trade deficit, border, FDI
    ("IN", "US"),   # India–USA: trade deal, tariffs, tech
    ("CN", "US"),   # China–USA: tariff war, decoupling, tech
    ("IN", "RU"),   # India–Russia: oil imports, sanctions, defense
    # New pairs
    ("IN", "PK"),   # India–Pakistan: border, terrorism, trade freeze
    ("IN", "IR"),   # India–Iran: oil, Chabahar port, sanctions
    ("IN", "IL"),   # India–Israel: defense, tech, Gaza conflict impact
    ("CN", "RU"),   # China–Russia: energy, sanctions bypass, alliance
    ("CN", "IR"),   # China–Iran: oil, BRI, sanctions
    ("CN", "PK"),   # China–Pakistan: CPEC, BRI, strategic
    ("US", "RU"),   # USA–Russia: sanctions, Ukraine, energy
    ("US", "IR"),   # USA–Iran: nuclear deal, sanctions, oil
    ("US", "IL"),   # USA–Israel: military aid, Gaza, diplomacy
    ("IL", "IR"),   # Israel–Iran: direct conflict, nuclear, proxies
    ("RU", "IR"),   # Russia–Iran: drones, energy, sanctions alignment
]
