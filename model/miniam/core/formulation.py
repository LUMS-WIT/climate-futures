"""Core LP formulation: a MESSAGE-style reference-energy-system model in linopy.

This generalizes the reference GAMS course model's two energy models — a toy
"simple model" and the full "world model", sourced from
https://github.com/volker-krey/ntnu_iam_2024 — into one reusable builder.
Equation names and structure follow the GAMS source directly so the code
reads next to
https://docs.messageix.org/en/stable/model/MESSAGE/model_core.html and next
to the GAMS itself.

A technology converts one or more (energy, level) commodities into others, with
a per-unit-activity ``input``/``output`` coefficient (GAMS: ``input``/``output``
tables). The simple model is the degenerate one-commodity case of this same
structure: a single (energy, level) pair with 1:1 output and no input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import linopy
import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class EnergySystemData:
    """Sets and parameters for one model instance. Field names match the GAMS
    identifiers so a reader can find the matching line in the ``.gms`` source.
    """

    technology: list[str]
    year: list[int]
    period_length: float
    energy: list[str]
    level: list[str]

    # commodity balance -------------------------------------------------
    energy_level: set[tuple[str, str]]
    """Valid (energy, level) combinations — GAMS ``energy_level(energy, level)``."""
    input: dict[tuple[str, str, str], float] = field(default_factory=dict)
    """(technology, energy, level) -> input coefficient."""
    output: dict[tuple[str, str, str], float] = field(default_factory=dict)
    """(technology, energy, level) -> output coefficient."""
    demand: dict[tuple[str, str, int], float] = field(default_factory=dict)
    """(energy, level, year) -> demand. Any GDP/beta growth is applied by the
    caller when building this table (see data/world_model.yaml) — the core only
    ever sees the resulting numbers, matching GAMS's own separation between
    parameter calculation and equation definition."""

    # technology economics ------------------------------------------------
    vom: dict[tuple[str, int], float] = field(default_factory=dict)
    """(technology, year) -> variable cost. GAMS ``vom``."""
    cost_capacity: dict[tuple[str, int], float] = field(default_factory=dict)
    """(technology, year) -> annuitized capacity cost. GAMS ``cost_capacity``,
    already computed by the caller (mirrors the GAMS parameter assignment,
    which happens before the equations are defined, not inside them)."""
    lifetime: dict[str, float] = field(default_factory=dict)
    hours: dict[str, float] = field(default_factory=dict)
    discount_rate: float = 0.05

    # emissions -------------------------------------------------------------
    co2_emission: dict[str, float] = field(default_factory=dict)
    """technology -> specific CO2 emission coefficient. GAMS ``CO2_emission``."""

    # diffusion ---------------------------------------------------------------
    diffusion_up: dict[str, float] = field(default_factory=dict)
    startup: dict[str, float] = field(default_factory=dict)

    # share constraints -----------------------------------------------------------
    tec_share: dict[str, list[str]] = field(default_factory=dict)
    """share name -> technologies on the constrained (left-hand) side."""
    tec_share_rhs: dict[str, list[str]] = field(default_factory=dict)
    """share name -> technologies on the reference (right-hand) side."""
    share_up: dict[str, float] = field(default_factory=dict)
    share_lo: dict[str, float] = field(default_factory=dict)

    # calibration -------------------------------------------------------------------
    act_fx: dict[tuple[str, int], float] = field(default_factory=dict)
    """(technology, year) -> fixed activity. GAMS ``ACT.FX``."""
    act_lo: dict[tuple[str, int], float] = field(default_factory=dict)
    """(technology, year) -> activity lower bound above the default 0. GAMS ``ACT.LO``."""
    cum_emiss_up: float | None = None
    emiss_up: dict[int, float] = field(default_factory=dict)


def _ord(years: list[int]) -> dict[int, int]:
    """1-indexed position of each year in the period sequence — GAMS ``ORD(year)``."""
    return {y: i + 1 for i, y in enumerate(years)}


def _eligible_vintage_mask(technology: list[str], year: list[int], period_length: float,
                            lifetime: dict[str, float]) -> xr.DataArray:
    """CAP_NEW(technology, vintage) is eligible to serve ACT(technology, year) iff
    ``vintage <= year`` and the vintage has not yet retired by ``year``.

    GAMS: ``(ORD(vintage) le ORD(year)) AND ((ORD(year) - ORD(vintage) + 1) *
    period_length le lifetime(technology))``. Technologies absent from
    ``lifetime`` (extraction/potential/grid-type flows in the world model) get an
    all-false mask — they have no capacity constraint at all, matching the
    GAMS ``$ (hours(technology) AND lifetime(technology))`` gate.
    """
    ordy = _ord(year)
    mask = np.zeros((len(technology), len(year), len(year)), dtype=bool)
    for ti, tech in enumerate(technology):
        life = lifetime.get(tech, 0)
        if not life:
            continue
        for yi, y in enumerate(year):
            for vi, v in enumerate(year):
                if ordy[v] <= ordy[y] and (ordy[y] - ordy[v] + 1) * period_length <= life:
                    mask[ti, yi, vi] = True
    return xr.DataArray(
        mask,
        dims=["technology", "year", "vintage"],
        coords={"technology": technology, "year": year, "vintage": year},
    )


def build_model(d: EnergySystemData) -> linopy.Model:
    """Build the LP. Mirrors ``energy_model_world.gms`` / ``simple_model.gms``
    equation-for-equation; see the docstring on each constraint block below for
    the corresponding GAMS equation name.
    """
    m = linopy.Model()

    tech_idx = pd.Index(d.technology, name="technology")
    year_idx = pd.Index(d.year, name="year")
    ordy = _ord(d.year)

    ACT = m.add_variables(lower=0, coords=[tech_idx, year_idx], name="ACT")
    CAP_NEW = m.add_variables(lower=0, coords=[tech_idx, year_idx], name="CAP_NEW")
    COST_ANNUAL = m.add_variables(coords=[year_idx], name="COST_ANNUAL")
    TOTAL_COST = m.add_variables(name="TOTAL_COST")
    EMISS = m.add_variables(coords=[year_idx], name="EMISS")
    CUM_EMISS = m.add_variables(name="CUM_EMISS")

    has_capacity = {t: bool(d.lifetime.get(t, 0)) and bool(d.hours.get(t, 0)) for t in d.technology}
    eligible = _eligible_vintage_mask(d.technology, d.year, d.period_length, d.lifetime)
    cap_new_vintage = CAP_NEW.rename({"year": "vintage"})

    # --- EQ_ENERGY_BALANCE(energy, level, year) ---------------------------------
    # supply (technology output minus input, summed over technology) >= demand,
    # for every valid (energy, level) combination.
    for energy, level in sorted(d.energy_level):
        supply_terms = []
        for tech in d.technology:
            coeff = d.output.get((tech, energy, level), 0.0) - d.input.get((tech, energy, level), 0.0)
            if coeff:
                supply_terms.append(coeff * ACT.sel(technology=tech))
        if not supply_terms:
            continue
        supply = sum(supply_terms[1:], start=supply_terms[0])
        rhs = xr.DataArray(
            [d.demand.get((energy, level, y), 0.0) for y in d.year],
            dims=["year"], coords={"year": d.year},
        )
        m.add_constraints(supply >= rhs, name=f"EQ_ENERGY_BALANCE_{energy}_{level}")

    # --- EQ_CAPACITY_BALANCE(technology, year) ----------------------------------
    # activity cannot exceed installed, not-yet-retired capacity times full-load hours.
    # only applies to technologies with both lifetime and hours defined.
    cap_techs = [t for t in d.technology if has_capacity[t]]
    if cap_techs:
        elig_sub = eligible.sel(technology=cap_techs)
        cap_sum = (elig_sub * cap_new_vintage.sel(technology=cap_techs)).sum("vintage")
        hours_da = xr.DataArray([d.hours[t] for t in cap_techs], dims=["technology"], coords={"technology": cap_techs})
        m.add_constraints(ACT.sel(technology=cap_techs) <= cap_sum * hours_da, name="EQ_CAPACITY_BALANCE")

    # --- EQ_EMISSION(year), EQ_EMISSION_CUMULATIVE ------------------------------
    emiss_coef = xr.DataArray(
        [d.co2_emission.get(t, 0.0) for t in d.technology], dims=["technology"], coords={"technology": d.technology},
    )
    m.add_constraints(EMISS == (ACT * emiss_coef).sum("technology"), name="EQ_EMISSION")
    m.add_constraints(CUM_EMISS == (EMISS * d.period_length).sum("year"), name="EQ_EMISSION_CUMULATIVE")
    if d.cum_emiss_up is not None:
        m.add_constraints(CUM_EMISS <= d.cum_emiss_up, name="EQ_CUM_EMISS_UP")
    for y, cap in d.emiss_up.items():
        m.add_constraints(EMISS.sel(year=y) <= cap, name=f"EQ_EMISS_UP_{y}")

    # --- EQ_DIFFUSION_UP(technology, year) --------------------------------------
    # new-capacity growth is bounded by last period's new capacity, plus a
    # startup constant that lets a technology at zero ever grow. Gated on
    # membership in diffusion_up, a different (larger, in the world model) set of
    # technologies than the capacity-balance one above — transcribed as-is.
    for yi in range(1, len(d.year)):
        y, y_prev = d.year[yi], d.year[yi - 1]
        for tech in d.technology:
            g = d.diffusion_up.get(tech)
            if g is None:
                continue
            growth = (1 + g) ** d.period_length
            su = d.startup.get(tech, 0.0)
            m.add_constraints(
                CAP_NEW.sel(technology=tech, year=y) <= CAP_NEW.sel(technology=tech, year=y_prev) * growth + su,
                name=f"EQ_DIFFUSION_UP_{tech}_{y}",
            )

    # --- EQ_SHARE_UP / EQ_SHARE_LO(share, year) ---------------------------------
    for share, lhs_techs in d.tec_share.items():
        rhs_techs = d.tec_share_rhs.get(share, [])
        lhs = ACT.sel(technology=lhs_techs).sum("technology")
        rhs = ACT.sel(technology=rhs_techs).sum("technology")
        if share in d.share_up:
            m.add_constraints(lhs <= rhs * d.share_up[share], name=f"EQ_SHARE_UP_{share}")
        if share in d.share_lo:
            m.add_constraints(lhs >= rhs * d.share_lo[share], name=f"EQ_SHARE_LO_{share}")

    # --- EQ_COST_ANNUAL(year), EQ_COST ------------------------------------------
    vom_da = xr.DataArray(
        [[d.vom.get((t, y), 0.0) for y in d.year] for t in d.technology],
        dims=["technology", "year"], coords={"technology": d.technology, "year": d.year},
    )
    capcost_da = xr.DataArray(
        [[d.cost_capacity.get((t, y), 0.0) for y in d.year] for t in d.technology],
        dims=["technology", "vintage"], coords={"technology": d.technology, "vintage": d.year},
    )
    activity_cost = ACT * vom_da
    capacity_cost = (eligible * capcost_da * cap_new_vintage).sum("vintage")
    per_tech_year_cost = activity_cost + capacity_cost
    m.add_constraints(COST_ANNUAL == per_tech_year_cost.sum("technology"), name="EQ_COST_ANNUAL")

    discount_factor = xr.DataArray(
        [(1 - d.discount_rate) ** (d.period_length * (ordy[y] - 1)) for y in d.year],
        dims=["year"], coords={"year": d.year},
    )
    m.add_constraints(
        TOTAL_COST == (COST_ANNUAL * d.period_length * discount_factor).sum("year"), name="EQ_COST"
    )

    # --- calibration: historical activity, ACT bounds ---------------------------
    for (tech, y), val in d.act_fx.items():
        m.add_constraints(ACT.sel(technology=tech, year=y) == val, name=f"ACT_FX_{tech}_{y}")
    for (tech, y), val in d.act_lo.items():
        m.add_constraints(ACT.sel(technology=tech, year=y) >= val, name=f"ACT_LO_{tech}_{y}")

    m.add_objective(TOTAL_COST)
    return m
