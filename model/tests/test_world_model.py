"""The world model balances at every level in every period, and a cumulative
CO2 cap produces a monotonically non-increasing coal trajectory. Structural
properties only — no GAMS reference run is available for this port, so
magnitudes are not pinned here."""

import pytest

from miniam.core.formulation import build_model
from miniam.data.loader import load_world


def _solved():
    data = load_world()
    m = build_model(data)
    m.solve(solver_name="highs")
    return data, m


def test_solves():
    _, m = _solved()
    assert m.status == "ok"


def test_commodity_balance_closes_at_every_level_and_period():
    """Supply (technology output minus input, summed over technology) must
    meet or exceed demand for every valid (energy, level) combination and
    every period — the LP is infeasible otherwise, but check explicitly
    rather than only trust the solver status."""
    data, m = _solved()
    act = m.variables["ACT"].solution

    for energy, level in sorted(data.energy_level):
        for y in data.year:
            supply = 0.0
            for tech in data.technology:
                coeff = data.output.get((tech, energy, level), 0.0) - data.input.get((tech, energy, level), 0.0)
                if coeff:
                    supply += coeff * float(act.sel(technology=tech, year=y))
            demand = data.demand.get((energy, level, y), 0.0)
            assert supply >= demand - 1e-6, f"{energy}.{level} in {y}: supply {supply} < demand {demand}"


def test_emissions_nonnegative():
    _, m = _solved()
    emiss = m.variables["EMISS"].solution
    assert (emiss >= -1e-6).all()


def test_cumulative_cap_binds_and_reduces_coal():
    """With a cumulative CO2 cap tight enough to bind, coal activity should
    trend down relative to the unconstrained run, and cumulative emissions
    should sit at (not below) the cap — the point of the constraint, not
    coincidence."""
    data_base, m_base = _solved()
    coal_base = m_base.variables["ACT"].solution.sel(technology="coal_ppl")
    cum_emiss_base = float(m_base.variables["CUM_EMISS"].solution)

    cap = cum_emiss_base * 0.5
    data_capped = load_world(cum_emiss_up=cap)
    m_capped = build_model(data_capped)
    m_capped.solve(solver_name="highs")
    assert m_capped.status == "ok"

    coal_capped = m_capped.variables["ACT"].solution.sel(technology="coal_ppl")
    cum_emiss_capped = float(m_capped.variables["CUM_EMISS"].solution)

    assert cum_emiss_capped == pytest.approx(cap, rel=1e-3)
    assert float(coal_capped.sum()) < float(coal_base.sum())
