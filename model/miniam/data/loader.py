"""Load an EnergySystemData instance from a YAML config file.

Parameters live in YAML so a dataset can be edited (technology added, cost
changed, a new period appended) without touching Python — see
``simple_model.yaml`` and ``world_model.yaml`` alongside this file. A regional
split would follow the same pattern with a ``regions.yaml`` aggregation map.

This module only holds two things Python must compute rather than store
directly: the capital-recovery-factor annuity behind ``cost_capacity``, and
(for the world model) GDP-scaled demand growth. Both mirror a GAMS parameter
assignment line-for-line — see the docstring on each function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from miniam.core.formulation import EnergySystemData

DATA_DIR = Path(__file__).parent


def _per_year(value: Any, years: list[int]) -> dict[int, float]:
    """A YAML scalar applies to every year; a YAML mapping gives per-year
    values directly (e.g. fom.wind_ppl's 2080 step-down in world_model.yaml)."""
    if isinstance(value, dict):
        return {int(y): float(v) for y, v in value.items()}
    return {y: float(value) for y in years}


def _crf(rate: float, life: float) -> float:
    """Capital recovery factor: annuitizes an investment cost over its
    lifetime at the given discount rate. GAMS:
    ``(1+r)^n * r / ((1+r)^n - 1)``."""
    return (1 + rate) ** life * rate / ((1 + rate) ** life - 1)


def load(path: str | Path, *, cost_capacity_scale: float = 1.0, cum_emiss_up: float | None = None) -> EnergySystemData:
    """Build an EnergySystemData from a YAML file.

    ``cost_capacity_scale`` covers the one real unit-conversion difference
    between the two reference datasets (world_model.yaml needs 1000x, per its
    own comment on why; simple_model.yaml needs 1x) — pass it explicitly
    rather than guess it from the file, since guessing wrong silently produces
    a solvable but wrong-by-a-constant-factor model.
    """
    raw = yaml.safe_load(Path(path).read_text())

    technology = raw["technology"]
    year = raw["year"]
    period_length = raw["period_length"]
    lifetime = {k: float(v) for k, v in raw.get("lifetime", {}).items()}
    hours = {k: float(v) for k, v in raw.get("hours", {}).items()}
    discount_rate = raw["discount_rate"]

    inv = raw.get("inv", {})
    fom = raw.get("fom", {})
    vom_raw = raw.get("vom", {})

    cost_capacity: dict[tuple[str, int], float] = {}
    for t in technology:
        life = lifetime.get(t, 0.0)
        hrs = hours.get(t, 0.0)
        if not life or not hrs:
            continue
        fom_by_year = _per_year(fom.get(t, 0.0), year)
        crf = _crf(discount_rate, life)
        for y in year:
            cost_capacity[(t, y)] = (inv.get(t, 0.0) * crf + fom_by_year[y]) * cost_capacity_scale

    vom = {(t, y): float(vom_raw.get(t, 0.0)) for t in technology for y in year}

    input_ = {(r["technology"], r["energy"], r["level"]): float(r["value"]) for r in raw.get("input", [])}
    output = {(r["technology"], r["energy"], r["level"]): float(r["value"]) for r in raw.get("output", [])}

    demand: dict[tuple[str, str, int], float] = {}
    if "demand_by_year" in raw:
        # explicit per-year demand table (simple_model.yaml)
        for r in raw["demand_by_year"]:
            demand[(r["energy"], r["level"], r["year"])] = float(r["value"])
    elif "demand_base" in raw:
        # base-year demand grown by GDP**beta (world_model.yaml) — GAMS:
        # demand(energy,level) * (gdp(year)/gdp(2020))**beta
        gdp = {int(y): float(v) for y, v in raw["gdp"].items()}
        beta = raw["gdp_beta"]
        base_year = year[0]
        for r in raw["demand_base"]:
            for y in year:
                demand[(r["energy"], r["level"], y)] = r["value"] * (gdp[y] / gdp[base_year]) ** beta

    diffusion_up = {k: float(v) for k, v in raw.get("diffusion_up", {}).items()}
    if "startup" in raw:
        startup = {k: float(v) for k, v in raw["startup"].items()}
    else:
        # world_model.yaml: one constant applies to every diffusion_up entry
        startup = {t: raw.get("startup_value", 0.0) for t in diffusion_up}

    act_fx = {(r["technology"], r["year"]): float(r["value"]) for r in raw.get("act_fx", [])}
    act_lo = {(r["technology"], r["year"]): float(r["value"]) for r in raw.get("act_lo", [])}

    return EnergySystemData(
        technology=technology,
        year=year,
        period_length=period_length,
        energy=raw["energy"],
        level=raw["level"],
        energy_level={tuple(pair) for pair in raw["energy_level"]},
        input=input_,
        output=output,
        demand=demand,
        vom=vom,
        cost_capacity=cost_capacity,
        lifetime=lifetime,
        hours=hours,
        discount_rate=discount_rate,
        co2_emission={k: float(v) for k, v in raw.get("co2_emission", {}).items()},
        diffusion_up=diffusion_up,
        startup=startup,
        tec_share=raw.get("tec_share", {}),
        tec_share_rhs=raw.get("tec_share_rhs", {}),
        share_up={k: float(v) for k, v in raw.get("share_up", {}).items()},
        share_lo={k: float(v) for k, v in raw.get("share_lo", {}).items()},
        act_fx=act_fx,
        act_lo=act_lo,
        cum_emiss_up=cum_emiss_up,
    )


def load_simple(**kwargs) -> EnergySystemData:
    return load(DATA_DIR / "simple_model.yaml", cost_capacity_scale=1.0, **kwargs)


def load_world(**kwargs) -> EnergySystemData:
    return load(DATA_DIR / "world_model.yaml", cost_capacity_scale=1000.0, **kwargs)
