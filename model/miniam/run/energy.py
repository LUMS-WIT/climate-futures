"""Driver for the energy-only LP. Mirrors ``model/run_energy.gms``: build,
solve, report. Runnable as a script or importable as a function, so the same
code backs both a notebook cell and a CLI call.

    python -m miniam.run.energy --dataset world
    python -m miniam.run.energy --dataset simple
"""

from __future__ import annotations

import argparse

import pandas as pd

from miniam.core.formulation import EnergySystemData, build_model


def solve(data: EnergySystemData, solver_name: str = "highs"):
    """Build and solve. Returns the solved linopy.Model — read ``.variables``
    off it for reporting (see ``report`` below)."""
    m = build_model(data)
    m.solve(solver_name=solver_name)
    return m


def report(m) -> dict[str, pd.DataFrame]:
    """Solution values as tidy DataFrames, one per reported variable —
    the Python equivalent of the CSV export at the bottom of
    energy_model_world.gms."""
    act = m.variables["ACT"].solution.to_dataframe(name="ACT").reset_index()
    cap_new = m.variables["CAP_NEW"].solution.to_dataframe(name="CAP_NEW").reset_index()
    emiss = m.variables["EMISS"].solution.to_dataframe(name="EMISS").reset_index()
    return {"ACT": act, "CAP_NEW": cap_new, "EMISS": emiss}


def binding_constraints(m, tol: float = 1e-6) -> pd.DataFrame:
    """Which constraints are binding (nonzero dual/shadow price) in each
    period — this is usually more instructive than the raw activity levels,
    since it shows what's actually driving the solution."""
    rows = []
    for name, con in m.constraints.items():
        dual = con.dual
        if dual is None:
            continue
        nonzero = dual.where(abs(dual) > tol, drop=True)
        for idx in nonzero.to_dataframe(name="dual").reset_index().to_dict("records"):
            rows.append({"constraint": name, **idx})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["world", "simple"], default="world")
    parser.add_argument("--cum-emiss-up", type=float, default=None,
                         help="optional cumulative CO2 cap (world dataset only)")
    args = parser.parse_args()

    from miniam.data.loader import load_simple, load_world

    if args.dataset == "world":
        data = load_world(cum_emiss_up=args.cum_emiss_up)
    else:
        data = load_simple()

    m = solve(data)
    print(f"status: {m.status}")
    print(f"TOTAL_COST: {m.variables['TOTAL_COST'].solution.item():.4f}")
    for name, df in report(m).items():
        print(f"\n{name}:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
