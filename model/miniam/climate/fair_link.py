"""Soft link between the energy LP and the real FaIR climate model.

Not implemented. This module only fixes the interface a future
``run/energy_climate.py`` needs.

Deliberately not a port of ``climate/FAIR-beta-4-3-1.gms``. That module is a
genuine NLP — its carbon-cycle reservoirs and thermal boxes take ``exp()`` and
``log()`` of decision variables — and linopy only solves LP/MILP/QP. The
maintained ``fair`` Python package is the right tool here instead of a linear
approximation of a nonlinear carbon cycle.

Intended contract:

    energy_result = miniam.run.energy.solve(data)
    emissions = extract_emissions_trajectory(energy_result)   # GtCO2/yr by year
    temperature = run_fair(emissions)                          # fair package call
    data = adjust_carbon_budget(data, temperature, target_c=2.0)
    energy_result = miniam.run.energy.solve(data)               # re-solve
    # repeat until temperature (or the budget adjustment) converges

The loop should converge in well under ten iterations, and cumulative
emissions under a 2C target should fall within FaIR's own expected range for
that target — not a number invented here.
"""

from __future__ import annotations


def run_fair(emissions_by_year: dict[int, float]):
    """Hand an emissions trajectory to the ``fair`` package and return its
    temperature projection. Requires the ``fair`` package (not yet a
    dependency of this project — add it only when this function is built)."""
    raise NotImplementedError("FaIR soft link not implemented yet.")


def adjust_carbon_budget(data, temperature_by_year: dict[int, float], target_c: float):
    """Given FaIR's temperature response, tighten or relax the energy model's
    cumulative-emissions bound (``EnergySystemData.cum_emiss_up``) toward the
    target. Returns a new ``EnergySystemData`` — inputs are not mutated in
    place, matching the rest of this codebase."""
    raise NotImplementedError("FaIR soft link not implemented yet.")
