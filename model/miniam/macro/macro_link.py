"""Iterative soft link between the energy LP and MACRO (CES production/utility).

Not implemented. This module only fixes the interface a future
``run/energy_macro.py`` needs.

Deliberately not a port of ``macro/macro_core.gms``. MACRO's production
function raises decision variables to fractional powers (``**(1/rho)``, a
nested CES form) and its utility function takes ``LOG(C(year))`` of a decision
variable — both nonlinear, and linopy only solves LP/MILP/QP. What
``message_ix`` does in practice, and what this module follows, is the
iterative link: energy and MACRO solve separately and exchange demands and
prices until convergence, which keeps the energy side linear throughout. A
hard-linked NLP variant, if ever wanted, is a separate Pyomo + IPOPT module —
not this one.

Intended contract:

    demands = initial_demand_guess(data)
    for _ in range(max_iterations):
        data = set_demand(data, demands)
        energy_result = miniam.run.energy.solve(data)
        prices = extract_shadow_prices(energy_result)   # EQ_ENERGY_BALANCE duals
        demands_new = macro_step(prices, demands)         # CES demand response
        if converged(demands, demands_new):
            break
        demands = demands_new

The loop should converge with higher energy service demand under a low
carbon price than under a high one — that comparison is the point of linking
MACRO in at all, and belongs in the test that exercises this once it exists.
"""

from __future__ import annotations


def macro_step(prices_by_sector: dict[str, float], demands_by_sector: dict[str, float]):
    """One MACRO-side iteration: given energy prices (the LP's own
    EQ_ENERGY_BALANCE shadow prices) and the previous demand guess, return an
    updated demand guess via the CES production/utility response. This is
    where macro_core.gms's math lives in Python form — not yet written."""
    raise NotImplementedError("MACRO iterative link not implemented yet.")


def converged(prev: dict[str, float], curr: dict[str, float], tol: float = 1e-4) -> bool:
    """Convergence check for the demand/price iteration loop above."""
    raise NotImplementedError("MACRO iterative link not implemented yet.")
