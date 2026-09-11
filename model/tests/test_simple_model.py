"""The 3-technology toy model solves, and the coal_power share constraint
binds at exactly 0.4."""

import pytest

from miniam.core.formulation import build_model
from miniam.data.loader import load_simple


def test_solves():
    data = load_simple()
    m = build_model(data)
    m.solve(solver_name="highs")
    assert m.status == "ok"


def test_coal_share_binds_at_point_four():
    data = load_simple()
    m = build_model(data)
    m.solve(solver_name="highs")

    act = m.variables["ACT"].solution
    for y in data.year:
        coal = float(act.sel(technology="coal_ppl", year=y))
        total = float(act.sel(year=y).sum())
        # share_up(coal_power) = 0.4 in simple_model.yaml — an upper bound,
        # but coal is the cheapest technology here (lowest vom, and no
        # diffusion advantage for gas/wind), so cost minimization should push
        # it to bind exactly at the cap.
        assert coal / total == pytest.approx(0.4, abs=1e-4)
