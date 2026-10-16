"""miniiam: a teaching-scale integrated assessment model with the structure of MESSAGEix.

ENGG562/ENV462 · Climate Change Governance: Science, Data & Models · LUMS

This module reimplements, in plain Python, the core of the MESSAGEix framework
(https://docs.messageix.org) at a scale small enough to read in one sitting:

- MESSAGE: a linear programme that finds the least-cost way of meeting energy-service
  demands with a network of technologies (the "reference energy system").
- MACRO: a single-sector neoclassical growth model that links energy costs to GDP
  and lets demand respond to energy prices.
- MESSAGE-MACRO: the iteration between the two, as implemented in
  message_ix/model/MESSAGE-MACRO_run.gms.

The public interface deliberately copies the names used by the ``message_ix`` package
(``Scenario``, ``make_df``, ``add_set``, ``add_par``, ``add_horizon``, ``solve``,
``var``, ``par``, ``clone``, ``add_macro``), and parameter and variable names follow
the MESSAGEix documentation (``input``, ``output``, ``inv_cost``, ``ACT``, ``CAP_NEW``,
``PRICE_COMMODITY`` and so on). Code written against this module can therefore be
moved to ``message_ix`` with few changes (see coding exercise 07).

Simplifications relative to MESSAGEix (stated here so that they are not forgotten):
one mode of operation, one time slice per year, parameters that vary by vintage are
averaged over vintages (activity is not tracked by vintage), no storage, no trade
between nodes, no land-use model, and CO2 as the only emission. The MESSAGE part is
solved with HiGHS through Pyomo; the MACRO part is solved with SciPy (SLSQP) instead
of CONOPT.

Requirements: numpy, pandas, scipy, pyomo, highspy.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

# =============================================================================
# 1. Parameter definitions (index columns identical to message_ix)
# =============================================================================
IDX = {
    "input": ["node_loc", "technology", "year_vtg", "year_act", "mode",
              "node_origin", "commodity", "level", "time", "time_origin"],
    "output": ["node_loc", "technology", "year_vtg", "year_act", "mode",
               "node_dest", "commodity", "level", "time", "time_dest"],
    "demand": ["node", "commodity", "level", "year", "time"],
    "inv_cost": ["node_loc", "technology", "year_vtg"],
    "fix_cost": ["node_loc", "technology", "year_vtg", "year_act"],
    "var_cost": ["node_loc", "technology", "year_vtg", "year_act", "mode", "time"],
    "capacity_factor": ["node_loc", "technology", "year_vtg", "year_act", "time"],
    "technical_lifetime": ["node_loc", "technology", "year_vtg"],
    "emission_factor": ["node_loc", "technology", "year_vtg", "year_act", "mode", "emission"],
    "historical_activity": ["node_loc", "technology", "year_act", "mode", "time"],
    "historical_new_capacity": ["node_loc", "technology", "year_vtg"],
    "growth_activity_up": ["node_loc", "technology", "year_act", "time"],
    "initial_activity_up": ["node_loc", "technology", "year_act", "time"],
    "growth_activity_lo": ["node_loc", "technology", "year_act", "time"],
    "bound_activity_up": ["node_loc", "technology", "year_act", "mode", "time"],
    "bound_activity_lo": ["node_loc", "technology", "year_act", "mode", "time"],
    "bound_emission": ["node", "type_emission", "type_tec", "type_year"],
    "tax_emission": ["node", "type_emission", "type_tec", "type_year"],
    "relation_activity": ["relation", "node_rel", "year_rel", "node_loc",
                          "technology", "year_act", "mode"],
    "relation_upper": ["relation", "node_rel", "year_rel"],
    "relation_lower": ["relation", "node_rel", "year_rel"],
    "interestrate": ["year"],
    "duration_period": ["year"],
}


def make_df(name, **data):
    """Return a DataFrame for parameter ``name`` with the message_ix column layout.

    Scalars are broadcast to the length of the longest list, as in ``message_ix.make_df``.
    Columns that are not given are left empty (``None``).
    """
    if name not in IDX:
        raise ValueError(f"Unknown parameter '{name}'. Known parameters: {sorted(IDX)}")
    cols = IDX[name] + ["value", "unit"]
    unknown = set(data) - set(cols)
    if unknown:
        raise ValueError(f"Columns {sorted(unknown)} are not dimensions of '{name}'")
    n = max([len(v) for v in data.values() if isinstance(v, (list, tuple, np.ndarray, pd.Series))] or [1])
    return pd.DataFrame({c: data.get(c, None) for c in cols}, index=range(n))


# =============================================================================
# 2. The Scenario container
# =============================================================================
class Scenario:
    """A model–scenario pair holding sets, parameters and (after solving) results.

    The interface mirrors ``message_ix.Scenario`` for the subset of features used here.
    """

    def __init__(self, model="miniIAM", scenario="baseline"):
        self.model, self.scenario = model, scenario
        self.sets = {k: [] for k in ("node", "technology", "commodity", "level", "mode",
                                     "emission", "year", "time", "relation")}
        self.sets["time"] = ["year"]
        self.cats = {}           # (set name, category) -> members, as Scenario.add_cat
        self.pars = {}           # parameter name -> DataFrame
        self.macro = None        # MACRO parameters after add_macro()
        self.solution = None     # results after solve()
        self.iterations = None   # MESSAGE-MACRO iteration log

    # -- sets ----------------------------------------------------------------
    def add_set(self, name, members):
        members = [members] if isinstance(members, (str, int)) else list(members)
        current = self.sets.setdefault(name, [])
        current += [m for m in members if m not in current]

    def set(self, name):
        return pd.Series(self.sets.get(name, []), name=name)

    def add_cat(self, name, cat, members):
        members = [members] if isinstance(members, (str, int)) else list(members)
        current = self.cats.setdefault((name, cat), [])
        current += [m for m in members if m not in current]

    def cat(self, name, cat):
        return list(self.cats.get((name, cat), []))

    def add_horizon(self, year, firstmodelyear=None):
        """Define the time horizon, as ``message_ix.Scenario.add_horizon``.

        The period duration of each year is the difference to the previous year; the
        first year takes the duration of the second.
        """
        year = sorted(int(y) for y in year)
        self.add_set("year", year)
        self.add_cat("year", "firstmodelyear", firstmodelyear or year[0])
        dur = np.diff(year).tolist()
        dur = [dur[0]] + dur
        self.add_par("duration_period", pd.DataFrame({"year": year, "value": dur, "unit": "y"}))

    @property
    def firstmodelyear(self):
        return self.cat("year", "firstmodelyear")[0]

    def model_years(self):
        return [y for y in sorted(self.sets["year"]) if y >= self.firstmodelyear]

    def years_active(self, node, tec, yr_vtg):
        """Periods in which capacity of vintage ``yr_vtg`` is still active
        (the technology's age at the end of the previous period is below its lifetime)."""
        yv = int(yr_vtg)
        tl = self.par("technical_lifetime", {"node_loc": node, "technology": tec})
        lt = float(tl[tl.year_vtg.astype(int) == yv]["value"].iloc[0])
        dur = self._dur()
        age, out = 0.0, []
        for y in sorted(self.sets["year"]):
            if y < yv:
                continue
            if age < lt:
                out.append(y)
            age += dur[y]
        return out

    def vintage_and_active_years(self, ya_args=None, tl_only=True, in_horizon=True):
        """Matched pairs of vintage and active periods, as in message_ix.

        With ``ya_args = (node, technology)``, only vintages with a technical_lifetime
        and active periods within that lifetime are returned. ``in_horizon=False``
        also returns active periods before the first model year.
        """
        years = sorted(self.sets["year"])
        if ya_args is None:
            pairs = [(v, a) for v in years for a in years if v <= a]
            ya_max = math.inf
        else:
            node, tec = ya_args[:2]
            tl = self.par("technical_lifetime", {"node_loc": node, "technology": tec})
            vintages = sorted(tl.year_vtg.astype(int).unique()) if len(ya_args) == 2 else [int(ya_args[2])]
            pairs = [(v, a) for v in vintages for a in self.years_active(node, tec, v)]
            ya_max = max(vintages) if tl_only else math.inf
        ya_min = self.firstmodelyear if in_horizon else -math.inf
        pairs = [(v, a) for v, a in pairs if ya_min <= a <= ya_max]
        return pd.DataFrame(pairs, columns=["year_vtg", "year_act"])

    def add_spatial_sets(self, data):
        """Add nodes, as message_ix.Scenario.add_spatial_sets({"country": name})."""
        for _, names in data.items():
            self.add_set("node", names)

    # -- parameters ----------------------------------------------------------
    def add_par(self, name, data, value=None, unit=None):
        """Add parameter data from a DataFrame, or (for one-dimensional parameters such
        as interestrate) from a list of keys with a single ``value`` and ``unit``."""
        if name not in IDX:
            raise ValueError(f"Unknown parameter '{name}'")
        if not isinstance(data, pd.DataFrame):
            if len(IDX[name]) != 1:
                raise ValueError(f"Pass a DataFrame for the {len(IDX[name])}-dimensional parameter '{name}'")
            keys = [data] if isinstance(data, (str, int)) else list(data)
            data = pd.DataFrame({IDX[name][0]: keys, "value": value, "unit": unit or "-"})
        data = data.copy()
        if "unit" not in data:
            data["unit"] = "-"
        new = pd.concat([self.pars.get(name), data], ignore_index=True)
        self.pars[name] = new.drop_duplicates(subset=IDX[name], keep="last").reset_index(drop=True)
        self.solution = None  # any change invalidates a previous solution

    def remove_par(self, name, data=None):
        if data is None:
            self.pars.pop(name, None)
        else:
            keys = IDX[name]
            cur = self.pars[name].merge(data[keys].drop_duplicates(), on=keys, how="left", indicator=True)
            self.pars[name] = cur[cur["_merge"] == "left_only"].drop(columns="_merge")
        self.solution = None

    def par(self, name, filters=None):
        df = self.pars.get(name, pd.DataFrame(columns=IDX[name] + ["value", "unit"])).copy()
        for col, vals in (filters or {}).items():
            vals = [vals] if isinstance(vals, (str, int)) else vals
            df = df[df[col].isin(vals)]
        return df.reset_index(drop=True)

    def clone(self, model=None, scenario=None, keep_solution=False):
        new = copy.deepcopy(self)
        new.model = model or self.model
        new.scenario = scenario or self.scenario
        if not keep_solution:
            new.solution, new.iterations = None, None
        return new

    # -- solving and results -------------------------------------------------
    def solve(self, model="MESSAGE", solver=None, convergence_criterion=0.01,
              max_adjustment=0.2, max_iteration=50, verbose=True, quiet=False, **kwargs):
        """Solve the scenario. ``model`` is "MESSAGE" or "MESSAGE-MACRO".
        ``quiet`` and other message_ix keyword arguments are accepted for compatibility."""
        verbose = verbose and not quiet
        if model == "MESSAGE":
            self.solution = solve_message(self, solver=solver)
        elif model == "MESSAGE-MACRO":
            if self.macro is None:
                raise RuntimeError("MACRO is not calibrated for this scenario: call add_macro() first.")
            solve_message_macro(self, convergence_criterion, max_adjustment, max_iteration,
                                solver=solver, verbose=verbose)
        else:
            raise ValueError("model must be 'MESSAGE' or 'MESSAGE-MACRO'")
        return self

    def has_solution(self):
        return self.solution is not None

    def var(self, name, filters=None):
        if self.solution is None:
            raise RuntimeError("The scenario has no solution: call solve() first.")
        if name not in self.solution:
            raise KeyError(f"Unknown variable '{name}'. Available: {sorted(self.solution)}")
        out = self.solution[name]
        if not isinstance(out, pd.DataFrame):
            return out
        out = out.copy()
        for col, vals in (filters or {}).items():
            vals = [vals] if isinstance(vals, (str, int)) else vals
            out = out[out[col].isin(vals)]
        return out.reset_index(drop=True)

    # message_ix keeps scenarios in a database and requires them to be checked out before
    # editing and committed afterwards. The mini model keeps everything in memory, so
    # these methods do nothing; they exist so that the same code runs on both.
    def check_out(self, *args, **kwargs):
        pass

    def commit(self, comment=""):
        pass

    def set_as_default(self):
        pass

    def add_macro(self, data, check_convergence=True, verbose=True):
        """Calibrate MACRO to this solved baseline and return a clone that carries it.

        ``data`` is a dict of DataFrames with the item names used by
        ``message_ix.Scenario.add_macro``: config, gdp_calibrate, price_ref, cost_ref,
        demand_ref, esub, drate, depr, kpvs, kgdp, lotol, aeei.
        """
        if self.solution is None:
            raise RuntimeError("Solve the baseline with MESSAGE before calibrating MACRO.")
        new = self.clone(keep_solution=True)
        new.macro = calibrate_macro(new, data, verbose=verbose)
        if check_convergence:
            new.solve("MESSAGE-MACRO", verbose=False)
            n = len(new.iterations)
            if verbose:
                print(f"Calibration check: MESSAGE-MACRO converged in {n} iteration(s)"
                      + (" (as expected for a calibrated baseline)." if n <= 2 else "."))
        return new

    # -- helpers used by the solver ------------------------------------------
    def _dur(self):
        d = self.par("duration_period")
        return dict(zip(d.year.astype(int), d.value.astype(float)))

    def __repr__(self):
        status = "solved" if self.solution is not None else "not solved"
        return f"<miniiam.Scenario {self.model}/{self.scenario} ({status})>"


# =============================================================================
# 3. MESSAGE: the energy-system optimisation (linear programme)
# =============================================================================
def _lookup(s, name, keys):
    """Average parameter values over the dimensions not listed in ``keys``."""
    df = s.par(name)
    if df.empty:
        return {}
    return df.groupby(keys)["value"].mean().to_dict()


def _solver(name=None):
    import pyomo.environ as pyo
    for cand in ([name] if name else ["appsi_highs", "highs", "cbc", "glpk"]):
        try:
            opt = pyo.SolverFactory(cand)
            if opt.available(exception_flag=False):
                return opt
        except Exception:
            continue
    raise RuntimeError("No LP solver found. Install HiGHS with: pip install highspy")


def build_message(s):
    """Build the MESSAGE linear programme for scenario ``s`` as a Pyomo model.

    Equation names follow the MESSAGEix documentation (Section "Mathematical formulation").
    """
    import pyomo.environ as pyo

    years = s.model_years()
    all_years = sorted(s.sets["year"])
    dur = s._dur()
    rate = _lookup(s, "interestrate", ["year"])
    r = {y: rate.get(y, 0.05) for y in all_years}
    first = years[0]
    last_end = years[-1]  # the model horizon ends with the last period
    prev = {y: all_years[all_years.index(y) - 1] for y in years}

    # Discount factors (df_year at the end of each period; df_period sums the years in it)
    base = first - dur[first]
    df_year = {y: (1 + r[y]) ** -(y - base) for y in years}
    df_period = {y: sum((1 + r[y]) ** -(y - k - base) for k in range(int(dur[y]))) for y in years}

    tec = list(s.sets["technology"])
    inp = _lookup(s, "input", ["technology", "year_act", "commodity", "level"])
    out = _lookup(s, "output", ["technology", "year_act", "commodity", "level"])
    dem = _lookup(s, "demand", ["commodity", "level", "year"])
    inv = _lookup(s, "inv_cost", ["technology", "year_vtg"])
    fix = _lookup(s, "fix_cost", ["technology", "year_act"])
    var = _lookup(s, "var_cost", ["technology", "year_act"])
    cf = _lookup(s, "capacity_factor", ["technology", "year_act"])
    life = _lookup(s, "technical_lifetime", ["technology", "year_vtg"])
    ef = _lookup(s, "emission_factor", ["technology", "year_act", "emission"])
    hist_act = _lookup(s, "historical_activity", ["technology", "year_act"])
    hist_cap = _lookup(s, "historical_new_capacity", ["technology", "year_vtg"])
    g_up = _lookup(s, "growth_activity_up", ["technology", "year_act"])
    i_up = _lookup(s, "initial_activity_up", ["technology", "year_act"])
    g_lo = _lookup(s, "growth_activity_lo", ["technology", "year_act"])
    b_up = _lookup(s, "bound_activity_up", ["technology", "year_act"])
    b_lo = _lookup(s, "bound_activity_lo", ["technology", "year_act"])

    # Technologies with a capacity: those with an investment cost, a capacity factor,
    # or historical capacity. Others (e.g. resource extraction) only have activity.
    cap_tec = sorted({t for (t, _) in inv} | {t for (t, _) in cf} | {t for (t, _) in hist_cap})
    lt = {}
    for t in cap_tec:
        default = np.mean([v for (tt, _), v in life.items() if tt == t] or [30])
        for v in all_years:
            lt[t, v] = life.get((t, v), default)

    def alive(t, v, y):
        return v <= y and y - v < lt[t, v]

    # End-of-horizon factor: share of the discounted lifetime that falls inside the horizon,
    # so that capacity built in late periods is not charged its full investment cost.
    def eoh(t, v):
        L = int(round(lt[t, v]))
        inside = max(0, min(L, last_end - v + int(dur[v])))
        rr = r[v]
        return sum((1 + rr) ** -k for k in range(inside)) / sum((1 + rr) ** -k for k in range(L))

    m = pyo.ConcreteModel(name=f"MESSAGE {s.model}/{s.scenario}")
    m.Y = pyo.Set(initialize=years, ordered=True)
    m.T = pyo.Set(initialize=tec, ordered=True)
    m.TC = pyo.Set(initialize=cap_tec, ordered=True)
    emissions = list(s.sets["emission"])
    m.E = pyo.Set(initialize=emissions, ordered=True)

    # ---- Variables ----
    m.ACT = pyo.Var(m.T, m.Y, within=pyo.NonNegativeReals)        # activity (e.g. GWa/yr)
    m.CAP_NEW = pyo.Var(m.TC, m.Y, within=pyo.NonNegativeReals)   # new capacity per year (GW/yr)
    m.EMISS = pyo.Var(m.E, m.Y, within=pyo.Reals)                 # emissions (Mt/yr)

    # CAPACITY_MAINTENANCE: installed capacity = surviving vintages of new capacity
    def cap_rule(m, t, y):
        hist = sum(dur[v] * hist_cap.get((t, v), 0.0) for v in all_years
                   if v < first and alive(t, v, y))
        new = sum(dur[v] * m.CAP_NEW[t, v] for v in years if alive(t, v, y))
        return hist + new
    m.CAP = pyo.Expression(m.TC, m.Y, rule=cap_rule)

    # ---- Equations ----
    # COMMODITY_BALANCE: supply - use >= demand, for every (commodity, level) and year
    flows = sorted({(c, l) for (_, _, c, l) in list(inp) + list(out)} | {(c, l) for (c, l, _) in dem})
    m.CL = pyo.Set(initialize=flows, dimen=2, ordered=True)

    def balance(m, c, l, y):
        supply = sum(v * m.ACT[t, y] for (t, ya, cc, ll), v in out.items() if (ya, cc, ll) == (y, c, l))
        use = sum(v * m.ACT[t, y] for (t, ya, cc, ll), v in inp.items() if (ya, cc, ll) == (y, c, l))
        return supply - use >= dem.get((c, l, y), 0.0)
    m.COMMODITY_BALANCE = pyo.Constraint(m.CL, m.Y, rule=balance)

    # CAPACITY_CONSTRAINT: activity cannot exceed what installed capacity can deliver
    m.CAPACITY_CONSTRAINT = pyo.Constraint(
        m.TC, m.Y, rule=lambda m, t, y: m.ACT[t, y] <= cf.get((t, y), 1.0) * m.CAP[t, y])

    # EMISSION_EQUIVALENCE: emissions are the sum of activity times emission factor
    m.EMISSION_EQUIVALENCE = pyo.Constraint(m.E, m.Y, rule=lambda m, e, y: m.EMISS[e, y] == sum(
        v * m.ACT[t, y] for (t, ya, ee), v in ef.items() if ya == y and ee == e))

    # ACTIVITY_BOUND_UP / _LO: absolute limits on activity (e.g. resource potentials)
    m.ACTIVITY_BOUND_UP = pyo.Constraint(
        [k for k in b_up if k[1] in years], rule=lambda m, t, y: m.ACT[t, y] <= b_up[t, y])
    m.ACTIVITY_BOUND_LO = pyo.Constraint(
        [k for k in b_lo if k[1] in years], rule=lambda m, t, y: m.ACT[t, y] >= b_lo[t, y])

    # ACTIVITY_CONSTRAINT_UP / _LO: dynamic constraints on how fast activity can grow
    # or decline from one period to the next (market penetration constraints)
    def act_prev(m, t, y):
        p = prev[y]
        return m.ACT[t, p] if p in years else hist_act.get((t, p), 0.0)

    def growth_up(m, t, y):
        g, n = g_up[t, y], dur[y]
        start = i_up.get((t, y), 0.0) * (((1 + g) ** n - 1) / g if g else n)
        return m.ACT[t, y] <= act_prev(m, t, y) * (1 + g) ** n + start
    m.ACTIVITY_CONSTRAINT_UP = pyo.Constraint([k for k in g_up if k[1] in years], rule=growth_up)
    m.ACTIVITY_CONSTRAINT_LO = pyo.Constraint(
        [k for k in g_lo if k[1] in years],
        rule=lambda m, t, y: m.ACT[t, y] >= act_prev(m, t, y) * (1 + g_lo[t, y]) ** dur[y])

    # RELATION_CONSTRAINT_UP / _LO: user-defined linear relations between activities
    rel_act = _lookup(s, "relation_activity", ["relation", "year_rel", "technology"])
    for kind, sense in (("relation_upper", "<="), ("relation_lower", ">=")):
        rhs = _lookup(s, kind, ["relation", "year_rel"])
        keys = [k for k in rhs if k[1] in years]
        name = "RELATION_CONSTRAINT_UP" if kind == "relation_upper" else "RELATION_CONSTRAINT_LO"

        def rel_rule(m, rel, y, rhs=rhs, sense=sense):
            lhs = sum(v * m.ACT[t, y] for (rr, yy, t), v in rel_act.items() if rr == rel and yy == y)
            return lhs <= rhs[rel, y] if sense == "<=" else lhs >= rhs[rel, y]
        setattr(m, name, pyo.Constraint(keys, rule=rel_rule))

    # EMISSION_CONSTRAINT: bounds on emissions; type_year "cumulative" bounds the
    # duration-weighted AVERAGE annual emissions over the horizon, as in MESSAGEix
    def emission_members(type_emission):
        return s.cat("emission", type_emission) or [type_emission]

    bounds = s.par("bound_emission")
    ec_keys, ec_rhs = [], {}
    for _, row in bounds.iterrows():
        k = (row.type_emission, str(row.type_year))
        ec_keys.append(k)
        ec_rhs[k] = float(row.value)
    total_dur = sum(dur[y] for y in years)

    def emission_rule(m, te, ty):
        es = emission_members(te)
        if ty == "cumulative":
            return sum(dur[y] * m.EMISS[e, y] for e in es for y in years) / total_dur <= ec_rhs[te, ty]
        return sum(m.EMISS[e, int(ty)] for e in es) <= ec_rhs[te, ty]
    m.EMISSION_CONSTRAINT = pyo.Constraint(ec_keys, rule=emission_rule)

    # Emission taxes (tax_emission): cost per unit of emission, by year
    tax = {}
    for _, row in s.par("tax_emission").iterrows():
        for e in emission_members(row.type_emission):
            tax[e, int(row.type_year)] = tax.get((e, int(row.type_year)), 0.0) + float(row.value)

    # COST_NODAL: undiscounted annual system cost in each period
    def cost_rule(m, y):
        return (sum(inv.get((t, y), 0.0) * eoh(t, y) * m.CAP_NEW[t, y] for t in cap_tec)
                + sum(fix.get((t, y), 0.0) * m.CAP[t, y] for t in cap_tec)
                + sum(var.get((t, y), 0.0) * m.ACT[t, y] for t in tec)
                + sum(tax.get((e, y), 0.0) * m.EMISS[e, y] for e in emissions))
    m.COST_NODAL = pyo.Expression(m.Y, rule=cost_rule)

    # OBJECTIVE: minimise the discounted sum of system costs
    m.OBJ = pyo.Objective(expr=sum(df_period[y] * m.COST_NODAL[y] for y in years), sense=pyo.minimize)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m._info = dict(years=years, dur=dur, df_period=df_period, df_year=df_year, tax=tax,
                   emissions=emissions, cap_tec=cap_tec, ec_keys=ec_keys, total_dur=total_dur,
                   emission_members=emission_members)
    return m


def solve_message(s, solver=None):
    """Solve MESSAGE and return the solution as a dict of DataFrames (message_ix names)."""
    import pyomo.environ as pyo
    from pyomo.opt import TerminationCondition as TC

    m = build_message(s)
    res = _solver(solver).solve(m, load_solutions=False)
    term = res.solver.termination_condition
    if term != TC.optimal:
        raise RuntimeError(
            f"MESSAGE did not solve to optimality (status: {term}). An infeasible model usually "
            "means that demand cannot be met under the bounds given: check emission bounds, "
            "growth constraints and resource limits.")
    m.solutions.load_from(res)

    info = m._info
    years, dur, dfp = info["years"], info["dur"], info["df_period"]
    v = pyo.value
    node = (s.sets["node"] or ["World"])[0]

    act = pd.DataFrame([(node, t, y, v(m.ACT[t, y])) for t in m.T for y in years],
                       columns=["node_loc", "technology", "year_act", "lvl"])
    cap_new = pd.DataFrame([(node, t, y, v(m.CAP_NEW[t, y])) for t in m.TC for y in years],
                           columns=["node_loc", "technology", "year_vtg", "lvl"])
    cap = pd.DataFrame([(node, t, y, v(m.CAP[t, y])) for t in m.TC for y in years],
                       columns=["node_loc", "technology", "year_act", "lvl"])
    emiss = pd.DataFrame([(node, e, y, v(m.EMISS[e, y])) for e in m.E for y in years],
                         columns=["node", "emission", "year", "lvl"])

    # PRICE_COMMODITY: the dual of the commodity balance, undiscounted (cost per unit)
    price = pd.DataFrame([(node, c, l, y, m.dual.get(m.COMMODITY_BALANCE[c, l, y], 0.0) / dfp[y])
                          for (c, l) in m.CL for y in years],
                         columns=["node", "commodity", "level", "year", "lvl"])

    # PRICE_EMISSION: the dual of the emission constraint expressed as an equivalent tax
    rows = []
    for (te, ty) in info["ec_keys"]:
        lam = -m.dual.get(m.EMISSION_CONSTRAINT[te, ty], 0.0)
        for y in years:
            if ty == "cumulative":
                rows.append((node, te, y, lam * dur[y] / info["total_dur"] / dfp[y]))
            elif int(ty) == y:
                rows.append((node, te, y, lam / dfp[y]))
    price_em = pd.DataFrame(rows, columns=["node", "type_emission", "year", "lvl"])

    # COST_NODAL_NET: system cost excluding emission-tax payments (a transfer, not a cost)
    cost = pd.DataFrame([(node, y, v(m.COST_NODAL[y])
                          - sum(info["tax"].get((e, y), 0.0) * v(m.EMISS[e, y]) for e in info["emissions"]))
                         for y in years], columns=["node", "year", "lvl"])
    demand = s.par("demand")[["node", "commodity", "level", "year", "value"]].rename(columns={"value": "lvl"})
    demand = demand[demand.year.isin(years)].reset_index(drop=True)

    return {"OBJ": v(m.OBJ), "ACT": act, "CAP": cap, "CAP_NEW": cap_new, "EMISS": emiss,
            "PRICE_COMMODITY": price, "PRICE_EMISSION": price_em, "COST_NODAL_NET": cost,
            "DEMAND": demand}


# =============================================================================
# 4. MACRO: the macro-economic growth model (non-linear programme)
# =============================================================================
def _item(data, name):
    """Scalar value of a single-row MACRO input item."""
    return float(data[name]["value"].iloc[0])


def calibrate_macro(s, data, max_it=100, tol=1e-4, verbose=True):
    """Calibrate MACRO to the solved baseline ``s``, following macro_calibration.gms.

    Production-function coefficients a and b are derived so that the base year
    reproduces the reference GDP, energy demand and energy prices. Then the potential
    GDP growth rates (grow) and the autonomous energy efficiency improvement rates
    (aeei) are adjusted, in alternating iterations, until MACRO, when given the baseline
    MESSAGE prices and costs, reproduces the baseline GDP and demand trajectories.
    """
    cfg = data["config"]
    sectors = list(cfg["sector"].dropna())
    mapping = {r.sector: (r.commodity, r.level) for r in cfg.dropna(subset=["sector"]).itertuples()}
    years = s.model_years()
    dur = s._dur()
    # As in message_ix, gdp_calibrate and cost_ref are in billion USD (G$); MACRO works in T$
    gdp = data["gdp_calibrate"].set_index("year")["value"].astype(float) / 1e3
    ym1 = max(y for y in gdp.index if y < years[0])                          # MACRO base year
    ym2 = max(y for y in gdp.index if y < ym1)

    esub, drate = _item(data, "esub"), _item(data, "drate")
    depr, kpvs, kgdp, lotol = (_item(data, k) for k in ("depr", "kpvs", "kgdp", "lotol"))
    rho = (esub - 1) / esub
    demand_ref = data["demand_ref"].set_index("sector")["value"].astype(float) / 1e3  # GWa -> TWa
    # Reference prices and cost in the base year. As in message_ix, when they are not
    # given they are extrapolated back from the first two model periods.
    base_link = message_to_macro(s, dict(mapping=mapping, years=years))
    if "price_ref" in data:
        price_ref = data["price_ref"].set_index("sector")["value"].astype(float) / 1e3  # $/kWa -> T$/TWa
    else:
        price_ref = pd.Series({sec: _extrapolate([base_link["eneprice"][sec, y] for y in years[:2]],
                                                 years[:2], ym1) for sec in sectors})
    if "cost_ref" in data:
        cost_ref = _item(data, "cost_ref") / 1e3                                      # G$ -> T$
    else:
        cost_ref = _extrapolate([base_link["total_cost"][y] for y in years[:2]], years[:2], ym1)

    gdp0 = float(gdp[ym1])
    k0 = kgdp * gdp0
    grow = {y: (gdp[y] / gdp[p]) ** (1 / dur[y]) - 1
            for p, y in zip([ym1] + years[:-1], years)}
    grow[ym1] = (gdp[ym1] / gdp[ym2]) ** (1 / dur[ym1]) - 1
    i0 = k0 * (grow[ym1] + depr)
    c0 = gdp0 - i0 - cost_ref

    # Production-function coefficients (calibrate.py: bconst, aconst)
    b = {sec: price_ref[sec] * (gdp0 / demand_ref[sec]) ** (rho - 1) for sec in sectors}
    a = (gdp0 ** rho - sum(b[sec] * demand_ref[sec] ** rho for sec in sectors)) / k0 ** (rho * kpvs)

    aeei = data["aeei"].set_index(["sector", "year"])["value"].astype(float).to_dict()
    aeei = {(sec, y): aeei.get((sec, y), 0.02) for sec in sectors for y in years}

    mac = dict(sectors=sectors, mapping=mapping, years=years, ym1=ym1, dur=dur, esub=esub,
               rho=rho, drate=drate, depr=depr, kpvs=kpvs, kgdp=kgdp, lotol=lotol, a=a, b=b,
               gdp0=gdp0, k0=k0, i0=i0, c0=c0, y0=gdp0, demand_ref=demand_ref.to_dict(),
               price_ref=price_ref.to_dict(), cost_ref=cost_ref, grow=grow, aeei=aeei,
               gdp_calibrate=gdp.to_dict())

    # Baseline MESSAGE information that MACRO is calibrated against
    link = message_to_macro(s, mac)
    target_dem = link["enestart"]

    for ctr in range(1, max_it + 1):
        sol = solve_macro(mac, link)
        dem_err = max(abs(sol["PHYSENE"][k] / target_dem[k] - 1) for k in target_dem)
        gdp_err = max(abs(sol["GDP"][y] / gdp[y] - 1) for y in years)
        if dem_err < tol and gdp_err < tol:
            break
        if ctr % 2 == 0:   # even iterations: correct potential GDP growth
            g_mac = {ym1: gdp0, **sol["GDP"]}
            seq = [ym1] + years
            for p, y in zip(seq[:-1], seq[1:]):
                grow[y] += ((gdp[y] / gdp[p]) ** (1 / dur[y])
                            - (g_mac[y] / g_mac[p]) ** (1 / dur[y]))
        else:              # odd iterations: correct the AEEI rates
            scale = {k: min(max(sol["PHYSENE"][k] / target_dem[k], 0.8), 1.2) for k in target_dem}
            for sec in sectors:
                prev_scale = 1.0
                for y in years:
                    aeei[sec, y] += (scale[sec, y] / prev_scale) ** (1 / dur[y]) - 1
                    prev_scale = scale[sec, y]
    mac["calibration"] = dict(iterations=ctr, demand_error=dem_err, gdp_error=gdp_err)
    if verbose:
        print(f"MACRO calibrated in {ctr} iterations "
              f"(max. demand error {dem_err:.1e}, max. GDP error {gdp_err:.1e}).")
    return mac


def _extrapolate(values, years, target):
    """Exponential extrapolation y = b * m**(x - x0) through two points."""
    (v1, v2), (x1, x2) = values, years
    m = (v2 / v1) ** (1 / (x2 - x1))
    return v1 * m ** (target - x1)


def _derived(mac):
    """Labour, new labour, utility discount factors and AEEI factors (macro_data_load.gms)."""
    years, dur, depr = mac["years"], mac["dur"], mac["depr"]
    seq = [mac["ym1"]] + years
    labor, udf, newlab, aeei_factor = {seq[0]: 1.0}, {seq[0]: 1.0}, {}, {}
    for sec in mac["sectors"]:
        aeei_factor[sec, seq[0]] = 1.0
    for p, y in zip(seq[:-1], seq[1:]):
        g = mac["grow"][y]
        labor[y] = labor[p] * (1 + g) ** dur[y]
        newlab[y] = max(labor[y] - labor[p] * (1 - depr) ** dur[y], 0.0) + 1e-9
        udf[y] = udf[p] * (1 - (mac["drate"] - g)) ** dur[y]
        for sec in mac["sectors"]:
            aeei_factor[sec, y] = aeei_factor[sec, p] * (1 - mac["aeei"][sec, y]) ** dur[y]
    fin_time = {y: abs(mac["drate"] - mac["grow"][y]) for y in years}
    return labor, newlab, udf, aeei_factor, fin_time


def message_to_macro(s, mac):
    """Information passed from a MESSAGE solution to MACRO: demand levels (enestart),
    shadow prices of demand (eneprice) and total energy-system cost (total_cost)."""
    dem = s.var("DEMAND").set_index(["commodity", "level", "year"])["lvl"]
    price = s.var("PRICE_COMMODITY").set_index(["commodity", "level", "year"])["lvl"]
    cost = s.var("COST_NODAL_NET").set_index("year")["lvl"]
    enestart, eneprice = {}, {}
    for sec, (c, l) in mac["mapping"].items():
        for y in mac["years"]:
            enestart[sec, y] = dem[c, l, y] / 1e3          # GWa -> TWa
            eneprice[sec, y] = max(price[c, l, y], 0) / 1e3  # $/kWa -> T$/TWa
    total_cost = {y: cost[y] / 1e6 for y in mac["years"]}    # M$ -> T$
    return dict(enestart=enestart, eneprice=eneprice, total_cost=total_cost)


def solve_macro(mac, link):
    """Solve MACRO for given MESSAGE information. Returns GDP, C, I, K, Y, EC and PHYSENE.

    The decision variables are investment I and physical energy use PHYSENE in each
    period. All other MACRO variables (KN, K, YN, Y, PRODENE, NEWENE, EC, C) follow from
    the equations of macro_core.gms, which are evaluated in order inside ``unpack``.
    """
    from scipy.optimize import minimize

    years, dur, sectors = mac["years"], mac["dur"], mac["sectors"]
    rho, kpvs, depr, a, b = mac["rho"], mac["kpvs"], mac["depr"], mac["a"], mac["b"]
    _, newlab, udf, aeei_factor, fin_time = _derived(mac)
    st, pr, tc = link["enestart"], link["eneprice"], link["total_cost"]
    n, S = len(years), len(sectors)

    def unpack(x):
        I = dict(zip(years, x[:n]))
        PHYS = {(sec, y): x[n + i * n + j] for i, sec in enumerate(sectors) for j, y in enumerate(years)}
        K_p, Y_p = mac["k0"], mac["y0"]
        PE_p = {sec: mac["demand_ref"][sec] for sec in sectors}
        out = {k: {} for k in ("K", "KN", "Y", "YN", "EC", "C", "NEWENE")}
        for y in years:
            keep = (1 - depr) ** dur[y]
            KN = dur[y] * I[y]                                        # NEW_CAPITAL
            K = K_p * keep + KN                                       # TOTAL_CAPITAL
            PE = {sec: PHYS[sec, y] / aeei_factor[sec, y] for sec in sectors}  # ENERGY_SUPPLY
            NE = {sec: PE[sec] - PE_p[sec] * keep for sec in sectors}         # NEW_ENERGY
            ces = a * max(KN, 1e-9) ** (rho * kpvs) * newlab[y] ** (rho * (1 - kpvs)) \
                + sum(b[sec] * max(NE[sec], 1e-9) ** rho for sec in sectors)
            YN = ces ** (1 / rho)                                     # NEW_PRODUCTION
            Y = Y_p * keep + YN                                       # TOTAL_PRODUCTION
            EC = tc[y] + sum(pr[sec, y] * (PHYS[sec, y] - st[sec, y])
                             + pr[sec, y] / st[sec, y] * (PHYS[sec, y] - st[sec, y]) ** 2
                             for sec in sectors)                      # COST_ENERGY
            C = Y - I[y] - EC                                         # CAPITAL_CONSTRAINT
            for k, val in (("K", K), ("KN", KN), ("Y", Y), ("YN", YN), ("EC", EC), ("C", C)):
                out[k][y] = val
            for sec in sectors:
                out["NEWENE"][sec, y] = NE[sec]
            K_p, Y_p, PE_p = K, Y, PE
        out["I"], out["PHYSENE"] = I, PHYS
        return out

    def utility(x):                                                   # UTILITY_FUNCTION
        o = unpack(x)
        u = 0.0
        for y in years:
            w = dur[y] + (1 / fin_time[y] if y == years[-1] else 0.0)
            u += udf[y] * math.log(max(o["C"][y], 1e-9)) * w
        return -u

    lot = mac["lotol"]

    def ineq(x):
        o = unpack(x)
        g = [o["C"][y] - lot * mac["c0"] for y in years]
        g += [o["NEWENE"][sec, y] - lot * st[sec, y] / aeei_factor[sec, y]
              for sec in sectors for y in years]
        yl = years[-1]                                                # TERMINAL_CONDITION
        g.append(o["I"][yl] - o["K"][yl] * (mac["grow"][yl] + depr))
        return np.array(g)

    x0 = np.array([mac["i0"] * mac["gdp_calibrate"].get(y, mac["y0"]) / mac["y0"] for y in years]
                  + [st[sec, y] for sec in sectors for y in years])
    x_last = mac.get("_x_last")  # warm start from the previous MACRO solution, if any
    if x_last is not None and len(x_last) == len(x0):
        x0 = x_last
    # The solver works on scaled variables z = x / scale, which are all close to 1
    scale = np.abs(x0) + 1e-9
    lower = np.array([lot * mac["i0"]] * n + [1e-6] * (n * S)) / scale

    import warnings

    def run(start):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # SLSQP clips steps to the bounds
            return minimize(lambda z: utility(z * scale), start, method="SLSQP",
                            bounds=[(lo, None) for lo in lower],
                            constraints=[{"type": "ineq", "fun": lambda z: ineq(z * scale)}],
                            options={"maxiter": 1000, "ftol": 1e-12})

    def acceptable(r):
        # SLSQP sometimes stops at the optimum with a line-search message; accept the
        # point if it satisfies all constraints
        return r.success or (np.all(ineq(r.x * scale) > -1e-7) and np.all(r.x >= lower - 1e-9))

    res = run(np.ones_like(x0))
    if not acceptable(res) and x_last is not None:   # retry from a cold start
        mac.pop("_x_last", None)
        return solve_macro(mac, link)
    if not acceptable(res):
        raise RuntimeError(f"MACRO did not converge: {res.message}")
    res.x = res.x * scale
    mac["_x_last"] = res.x
    o = unpack(res.x)
    o["GDP"] = {y: o["I"][y] + o["C"][y] + o["EC"][y] for y in years}
    o["UTILITY"] = -res.fun
    return o


# =============================================================================
# 5. MESSAGE-MACRO: the iteration between the two models
# =============================================================================
def solve_message_macro(s, convergence_criterion=0.01, max_adjustment=0.2,
                        max_iteration=50, solver=None, verbose=True):
    """Iterate MESSAGE and MACRO until demands converge (MESSAGE-MACRO_run.gms).

    1. Solve MESSAGE with the current demands.
    2. Pass demand levels, demand prices and system cost to MACRO and solve it.
    3. Rescale MESSAGE demands by MACRO's physical energy use, limited to
       +/- max_adjustment per iteration (reduced when oscillation is detected).
    4. Stop when the largest relative demand adjustment is below convergence_criterion.
    """
    mac = s.macro
    base_demand = s.par("demand")
    s.pars["demand"] = base_demand.copy()
    scaling, prev_adj, log = 1.0, None, []
    for it in range(1, max_iteration + 1):
        sol = solve_message(s, solver=solver)
        s.solution = sol
        link = message_to_macro(s, mac)
        mo = solve_macro(mac, link)

        scale = {}
        for k in link["enestart"]:
            raw = mo["PHYSENE"][k] / link["enestart"][k]
            scale[k] = min(max(raw, 1 - max_adjustment / scaling), 1 + max_adjustment / scaling)
        pos = max([v - 1 for v in scale.values() if v > 1] or [0.0])
        neg = min([v - 1 for v in scale.values() if v < 1] or [0.0])
        adj = neg if neg < -pos else pos
        log.append(dict(iteration=it, OBJ=sol["OBJ"], max_adjustment=adj,
                        GDP_2050=mo["GDP"].get(2050, np.nan), scaling=scaling))
        if verbose:
            print(f"iteration {it:2d}: MESSAGE objective {sol['OBJ']:,.0f}  "
                  f"largest demand adjustment {adj:+.4f}")
        if abs(adj) < convergence_criterion:
            break
        if prev_adj is not None and np.sign(prev_adj) == -np.sign(adj) and abs(prev_adj) > abs(adj) * 0.9:
            scaling *= math.sqrt(2)  # oscillation detected: take smaller steps
        prev_adj = adj

        # Apply the demand adjustment for the next MESSAGE solve
        dem = s.pars["demand"]
        for sec, (c, l) in mac["mapping"].items():
            for y in mac["years"]:
                mask = (dem.commodity == c) & (dem.level == l) & (dem.year == y)
                dem.loc[mask, "value"] = dem.loc[mask, "value"] * scale[sec, y]
    else:
        s.pars["demand"] = base_demand
        raise RuntimeError(f"MESSAGE-MACRO did not converge in {max_iteration} iterations.")
    s.pars["demand"] = base_demand  # the input data are unchanged; DEMAND reports the result

    # Report MACRO results alongside the MESSAGE solution. As in message_ix, GDP is
    # reported in billion USD (G$) and the MACRO variables C, I, K, EC in trillion USD.
    sol = s.solution
    node = (s.sets["node"] or ["World"])[0]
    sol["GDP"] = pd.DataFrame([(node, y, 1e3 * mo["GDP"][y]) for y in mac["years"]], columns=["node", "year", "lvl"])
    for k in ("C", "I", "K", "EC"):
        sol[k] = pd.DataFrame([(node, y, mo[k][y]) for y in mac["years"]], columns=["node", "year", "lvl"])
    sol["PHYSENE"] = pd.DataFrame([(node, sec, y, v) for (sec, y), v in mo["PHYSENE"].items()],
                                  columns=["node", "sector", "year", "lvl"])
    s.iterations = pd.DataFrame(log)
    return s


# =============================================================================
# 6. Reporting helpers
# =============================================================================
TCRE = 0.45 / 1000  # °C per GtCO2: IPCC AR6 WGI best estimate (likely range 0.27–0.63 per 1000 GtCO2)


def cumulative_co2(s, emission="CO2", start_year=None):
    """Cumulative CO2 emissions (GtCO2) from the start of the first period, by year."""
    em = s.var("EMISS")
    em = em[em.emission == emission].set_index("year")["lvl"].sort_index()
    dur = s._dur()
    return (em * pd.Series(dur).reindex(em.index)).cumsum() / 1e3  # Mt -> Gt


def warming(s, t0=1.1, emission="CO2"):
    """Approximate global warming (°C above 1850–1900) using the TCRE relationship.

    t0 is observed warming at the start of the horizon. Non-CO2 forcing is ignored,
    which is the main limitation of this estimate.
    """
    return t0 + TCRE * cumulative_co2(s, emission)


def timeseries(s, var, by="technology", filters=None, value="lvl"):
    """Pivot a result variable into a year x category table."""
    df = s.var(var)
    for col, vals in (filters or {}).items():
        vals = [vals] if isinstance(vals, (str, int)) else vals
        df = df[df[col].isin(vals)]
    year = "year_act" if "year_act" in df else ("year_vtg" if "year_vtg" in df else "year")
    return df.pivot_table(index=year, columns=by, values=value, aggfunc="sum").fillna(0.0)



# IAMC variable names for reporting (the format of the AR6 Scenarios Database)
IAMC_ELECTRICITY = {
    "coal_ppl": "Secondary Energy|Electricity|Coal|w/o CCS",
    "coal_ccs": "Secondary Energy|Electricity|Coal|w/ CCS",
    "gas_ppl": "Secondary Energy|Electricity|Gas|w/o CCS",
    "gas_ccs": "Secondary Energy|Electricity|Gas|w/ CCS",
    "nuc_ppl": "Secondary Energy|Electricity|Nuclear",
    "hydro_ppl": "Secondary Energy|Electricity|Hydro",
    "wind_ppl": "Secondary Energy|Electricity|Wind",
    "solar_pv": "Secondary Energy|Electricity|Solar",
    "bio_ppl": "Secondary Energy|Electricity|Biomass|w/o CCS",
    "bio_ccs": "Secondary Energy|Electricity|Biomass|w/ CCS",
}


def to_iamc(s, region=None):
    """Main results of a solved World scenario as a long table in the IAMC format
    (model, scenario, region, variable, unit, year, value), ready for pyam."""
    region = region or (s.sets["node"] or ["World"])[0]
    rows = []
    em = s.var("EMISS")
    for _, r in em.iterrows():
        rows.append(("Emissions|CO2", "Mt CO2/yr", r.year, r.lvl))
    act = s.var("ACT")
    for _, r in act[act.technology.isin(IAMC_ELECTRICITY)].iterrows():
        rows.append((IAMC_ELECTRICITY[r.technology], "EJ/yr", r.year_act, r.lvl * 0.031536))
    elec = act[act.technology.isin(IAMC_ELECTRICITY)].groupby("year_act")["lvl"].sum() * 0.031536
    rows += [("Secondary Energy|Electricity", "EJ/yr", y, v) for y, v in elec.items()]
    pe = s.var("PRICE_EMISSION")
    rows += [("Price|Carbon", "US$/t CO2", r.year, r.lvl) for _, r in pe.iterrows()]
    if "GDP" in s.solution:
        rows += [("GDP|MER", "billion US$/yr", r.year, r.lvl) for _, r in s.var("GDP").iterrows()]
    df = pd.DataFrame(rows, columns=["variable", "unit", "year", "value"])
    df.insert(0, "region", region)
    df.insert(0, "scenario", s.scenario)
    df.insert(0, "model", s.model)
    return df

# =============================================================================
# 7. Example datasets
# =============================================================================
# Both builders only use make_df(), add_set(), add_cat(), add_horizon() and add_par(),
# so they work unchanged on a miniiam.Scenario and on a message_ix.Scenario.

UNITS = ["GWa", "GW/y", "USD/kW", "USD/kWa", "tCO2/kWa", "MtCO2", "USD/tCO2", "G$", "T$"]


def _register_units(scen):
    """message_ix checks units against its database: register the ones used here."""
    platform = getattr(scen, "platform", None)
    if platform is not None:
        for u in UNITS:
            platform.add_unit(u, comment="used by miniiam example data")


def _add_node(scen, node):
    """Add a node. In message_ix the node "World" exists already, and other nodes are
    added with add_spatial_sets()."""
    if hasattr(scen, "platform"):
        if node not in list(scen.set("node")):
            scen.add_spatial_sets({"country": node})
    else:
        scen.add_set("node", node)


def _va_pairs(years, model_years, life=None):
    """(year_vtg, year_act) pairs: every vintage still alive in each model year."""
    if life is None:
        return [(y, y) for y in model_years]
    return [(v, a) for a in model_years for v in years if v <= a and a - v < life]


def _add_tec(scen, node, t, va, out, inp=None, unit="GWa"):
    """Add output (and input) rows for technology t. out/inp = (commodity, level, value)."""
    yv, ya = [p[0] for p in va], [p[1] for p in va]
    common = dict(node_loc=node, technology=t, year_vtg=yv, year_act=ya, mode="standard",
                  time="year", unit=unit)
    c, l, v = out
    scen.add_par("output", make_df("output", node_dest=node, time_dest="year",
                                   commodity=c, level=l, value=v, **common))
    if inp is not None:
        c, l, v = inp
        scen.add_par("input", make_df("input", node_origin=node, time_origin="year",
                                      commodity=c, level=l, value=v, **common))


# -----------------------------------------------------------------------------
# 7a. Westeros: the teaching example of the message_ix tutorials
# -----------------------------------------------------------------------------
# Values are those of message_ix/tutorial/westeros/westeros_baseline.ipynb, so that
# results can be compared directly with the official tutorial (coding exercise 07).

WESTEROS = dict(
    history=[690], horizon=[700, 710, 720],
    demand_per_year=40 * 12 * 1000 / 8760,      # GWa: as in the tutorial
    gdp_profile=[1.0, 1.5, 1.9],
    grid_efficiency=0.9, coal_fraction=0.6,
    lifetime={"coal_ppl": 20, "wind_ppl": 20, "bulb": 1, "grid": 30},
    capacity_factor={"coal_ppl": 1, "wind_ppl": 0.36, "bulb": 1, "grid": 1},
    inv_cost={"coal_ppl": 500, "wind_ppl": 1500, "bulb": 5, "grid": 800},
    fix_cost={"coal_ppl": 30, "wind_ppl": 10, "grid": 16},
    var_cost={"coal_ppl": 30},
    growth={"coal_ppl": 0.1, "wind_ppl": 0.1},
    coal_emission_factor=7.4,                    # tCO2/kWa, as in westeros_emissions_bounds
)


def build_westeros(scen, node="Westeros", emissions=True):
    """Populate ``scen`` with the Westeros baseline of the message_ix tutorials."""
    _register_units(scen)
    w = WESTEROS
    years, horizon = w["history"] + w["horizon"], w["horizon"]
    scen.add_horizon(year=years, firstmodelyear=horizon[0])
    _add_node(scen, node)
    scen.add_set("commodity", ["electricity", "light"])
    scen.add_set("level", ["secondary", "final", "useful"])
    scen.add_set("technology", ["coal_ppl", "wind_ppl", "grid", "bulb"])
    scen.add_set("mode", "standard")

    demand = [float(round(w["demand_per_year"] * g)) for g in w["gdp_profile"]]
    scen.add_par("demand", make_df("demand", node=node, commodity="light", level="useful",
                                   year=horizon, time="year", value=demand, unit="GWa"))
    for t, life in w["lifetime"].items():
        scen.add_par("technical_lifetime", make_df("technical_lifetime", node_loc=node,
                                                   technology=t, year_vtg=years, value=life, unit="y"))
    # All surviving (vintage, active) pairs, including the historical period, as returned by
    # message_ix: scenario.vintage_and_active_years((node, tec), in_horizon=False)
    dur = 10
    va = {t: [(v, a) for v in years for a in years if v <= a and a - v < max(life, dur)]
          for t, life in w["lifetime"].items()}

    _add_tec(scen, node, "bulb", va["bulb"], ("light", "useful", 1.0), ("electricity", "final", 1.0), unit="-")
    _add_tec(scen, node, "grid", va["grid"], ("electricity", "final", 1.0),
             ("electricity", "secondary", 1.0 / w["grid_efficiency"]), unit="-")
    _add_tec(scen, node, "coal_ppl", va["coal_ppl"], ("electricity", "secondary", 1.0), unit="-")
    _add_tec(scen, node, "wind_ppl", va["wind_ppl"], ("electricity", "secondary", 1.0), unit="-")

    for t, cf in w["capacity_factor"].items():
        scen.add_par("capacity_factor", make_df(
            "capacity_factor", node_loc=node, technology=t, year_vtg=[p[0] for p in va[t]],
            year_act=[p[1] for p in va[t]], time="year", value=cf, unit="-"))
    for t, g in w["growth"].items():
        scen.add_par("growth_activity_up", make_df("growth_activity_up", node_loc=node, technology=t,
                                                   year_act=horizon, time="year", value=g, unit="-"))

    historic_generation = 0.5 * w["demand_per_year"] / w["grid_efficiency"]
    old_activity = {"coal_ppl": w["coal_fraction"] * historic_generation,
                    "wind_ppl": (1 - w["coal_fraction"]) * historic_generation,
                    "grid": historic_generation}
    for t, val in old_activity.items():
        scen.add_par("historical_activity", make_df(
            "historical_activity", node_loc=node, technology=t, year_act=w["history"],
            mode="standard", time="year", value=val, unit="GWa"))
        scen.add_par("historical_new_capacity", make_df(
            "historical_new_capacity", node_loc=node, technology=t, year_vtg=w["history"],
            value=val / (10 * w["capacity_factor"][t]), unit="GW/y"))

    scen.add_par("interestrate", make_df("interestrate", year=horizon, value=0.05, unit="-"))
    for t, val in w["inv_cost"].items():
        scen.add_par("inv_cost", make_df("inv_cost", node_loc=node, technology=t,
                                         year_vtg=horizon, value=val, unit="USD/kW"))
    for t, val in w["fix_cost"].items():
        scen.add_par("fix_cost", make_df("fix_cost", node_loc=node, technology=t,
                                         year_vtg=[p[0] for p in va[t]], year_act=[p[1] for p in va[t]],
                                         value=val, unit="USD/kWa"))
    for t, val in w["var_cost"].items():
        scen.add_par("var_cost", make_df("var_cost", node_loc=node, technology=t,
                                         year_vtg=[p[0] for p in va[t]], year_act=[p[1] for p in va[t]],
                                         mode="standard", time="year", value=val, unit="USD/kWa"))
    if emissions:
        scen.add_set("emission", "CO2")
        scen.add_cat("emission", "GHG", "CO2")
        scen.add_par("emission_factor", make_df(
            "emission_factor", node_loc=node, technology="coal_ppl",
            year_vtg=[p[0] for p in va["coal_ppl"]], year_act=[p[1] for p in va["coal_ppl"]],
            mode="standard", emission="CO2", value=w["coal_emission_factor"], unit="tCO2/kWa"))
    return scen


def westeros_macro_data(node="Westeros"):
    """MACRO input data for Westeros: message_ix/tests/data/westeros_macro_input.xlsx."""
    one = lambda v: pd.DataFrame({"node": [node], "value": [v], "unit": ["-"]})
    return {
        "config": pd.DataFrame({"node": [node, None, None], "sector": ["light", None, None],
                                "commodity": ["light", None, None], "level": ["useful", None, None],
                                "year": [700, 710, 720]}),
        # The test file labels these values "T$", but MESSAGE-MACRO reads them as G$
        "gdp_calibrate": pd.DataFrame({"node": node, "year": [680, 690, 700, 710, 720],
                                       "value": [350, 500, 1000, 2000, 3000], "unit": "G$"}),
        "price_ref": pd.DataFrame({"node": [node], "sector": ["light"], "value": [211], "unit": "USD/kWa"}),
        "cost_ref": pd.DataFrame({"node": [node], "value": [6.18242], "unit": ["G$"]}),
        "demand_ref": pd.DataFrame({"node": [node], "sector": ["light"], "value": [27], "unit": "GWa"}),
        "esub": one(0.2), "drate": one(0.05), "depr": one(0.05), "kpvs": one(0.3),
        "kgdp": one(3.0), "lotol": one(0.05),
        "aeei": pd.DataFrame({"node": node, "sector": "light", "year": [700, 710, 720], "value": 0.02, "unit": "-"}),
    }


# -----------------------------------------------------------------------------
# 7b. World: a global reference energy system with the same structure
# -----------------------------------------------------------------------------
# All values are rounded, illustrative assumptions of a plausible order of magnitude for
# the world energy system around 2020. They are NOT a calibrated dataset and should not
# be cited as such. Units follow MESSAGEix conventions:
#   activity GWa/yr (1 GWa = 8.76 TWh = 0.0315 EJ), capacity GW,
#   inv_cost $/kW, fix_cost $/kW/yr, var_cost $/kWa, emissions MtCO2.

WORLD_YEARS = [1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
WORLD_FIRST_YEAR = 2030

# CO2 emission factors of fuels: IPCC 2006 default values (kg CO2/GJ) x 31.536 GJ/kWa
EF = {"coal": 94.6 * 31.536 / 1e3, "gas": 56.1 * 31.536 / 1e3,
      "oil": 73.3 * 31.536 / 1e3, "biomass": 100.0 * 31.536 / 1e3}   # tCO2/kWa = MtCO2/GWa
CAPTURE = 0.9  # share of CO2 captured by CCS plants

# Technology table: (input commodity, input level, output commodity, output level, efficiency)
WORLD_TECH = {
    # resource extraction and fuel processing
    "coal_extr":  (None, None, "coal", "primary", 1.0),
    "gas_extr":   (None, None, "gas", "primary", 1.0),
    "oil_extr":   (None, None, "crudeoil", "primary", 1.0),
    "bio_extr":   (None, None, "biomass", "primary", 1.0),
    "oil_ref":    ("crudeoil", "primary", "oil", "secondary", 0.90),
    # electricity generation
    "coal_ppl":   ("coal", "primary", "electricity", "secondary", 0.37),
    "coal_ccs":   ("coal", "primary", "electricity", "secondary", 0.30),
    "gas_ppl":    ("gas", "primary", "electricity", "secondary", 0.52),
    "gas_ccs":    ("gas", "primary", "electricity", "secondary", 0.45),
    "nuc_ppl":    (None, None, "electricity", "secondary", 1.0),
    "hydro_ppl":  (None, None, "electricity", "secondary", 1.0),
    "wind_ppl":   (None, None, "electricity", "secondary", 1.0),
    "solar_pv":   (None, None, "electricity", "secondary", 1.0),
    "bio_ppl":    ("biomass", "primary", "electricity", "secondary", 0.30),
    "bio_ccs":    ("biomass", "primary", "electricity", "secondary", 0.25),
    # transmission and distribution (the "grid" of Westeros), hydrogen production
    "elec_t_d":   ("electricity", "secondary", "electricity", "final", 0.90),
    "h2_elec":    ("electricity", "final", "hydrogen", "final", 0.65),
    # end use: specific electricity (the "bulb" of Westeros: appliances, motors, lighting, ICT)
    "elec_spec":  ("electricity", "final", "spec", "useful", 1.0),
    # end use: thermal energy in industry and buildings
    "coal_i":     ("coal", "primary", "therm", "useful", 0.70),
    "gas_i":      ("gas", "primary", "therm", "useful", 0.85),
    "oil_i":      ("oil", "secondary", "therm", "useful", 0.80),
    "bio_i":      ("biomass", "primary", "therm", "useful", 0.50),
    "elec_i":     ("electricity", "final", "therm", "useful", 1.50),
    "h2_i":       ("hydrogen", "final", "therm", "useful", 0.90),
    # end use: transport
    "oil_trp":    ("oil", "secondary", "transport", "useful", 0.25),
    "gas_trp":    ("gas", "primary", "transport", "useful", 0.25),
    "bio_trp":    ("biomass", "primary", "transport", "useful", 0.15),
    "elec_trp":   ("electricity", "final", "transport", "useful", 0.75),
    "h2_trp":     ("hydrogen", "final", "transport", "useful", 0.45),
}

# Technologies with a capacity: inv_cost by year (interpolated linearly), fix_cost,
# var_cost (non-fuel operating cost), capacity factor and technical lifetime
WORLD_CAPACITY = {
    "coal_ppl":  dict(inv={2020: 1600}, fix=45, var=35, cf=0.85, life=40),
    "coal_ccs":  dict(inv={2020: 4500, 2050: 3500}, fix=90, var=60, cf=0.85, life=40),
    "gas_ppl":   dict(inv={2020: 900}, fix=25, var=25, cf=0.85, life=30),
    "gas_ccs":   dict(inv={2020: 2200, 2050: 1700}, fix=55, var=45, cf=0.85, life=30),
    "nuc_ppl":   dict(inv={2020: 5500}, fix=120, var=90, cf=0.85, life=60),
    "hydro_ppl": dict(inv={2020: 2600}, fix=50, var=0, cf=0.42, life=80),
    "wind_ppl":  dict(inv={2020: 1500, 2030: 1250, 2050: 1050, 2100: 950}, fix=40, var=0, cf=0.32, life=25),
    "solar_pv":  dict(inv={2020: 1100, 2030: 700, 2050: 450, 2100: 380}, fix=15, var=0, cf=0.18, life=25),
    "bio_ppl":   dict(inv={2020: 2500}, fix=80, var=40, cf=0.80, life=30),
    "bio_ccs":   dict(inv={2020: 5000, 2050: 4000}, fix=130, var=70, cf=0.80, life=30),
    "h2_elec":   dict(inv={2020: 1800, 2030: 1000, 2050: 600, 2100: 500}, fix=40, var=10, cf=0.50, life=20),
}

# Variable costs of technologies without capacity ($/kWa of output). For extraction this
# is the fuel cost; for end-use technologies it stands in for the levelised cost of the
# end-use equipment (vehicles, boilers, heat pumps) relative to a conventional option.
WORLD_VAR_COST = {
    "coal_extr": {2020: 80}, "gas_extr": {2020: 190}, "oil_extr": {2020: 300}, "bio_extr": {2020: 160},
    "oil_ref": {2020: 30}, "elec_t_d": {2020: 120}, "elec_spec": {2020: 0},
    "coal_i": {2020: 20}, "gas_i": {2020: 20}, "oil_i": {2020: 20}, "bio_i": {2020: 30},
    "elec_i": {2020: 120, 2050: 70}, "h2_i": {2020: 120, 2050: 60},
    "oil_trp": {2020: 100}, "gas_trp": {2020: 800}, "bio_trp": {2020: 300},
    "elec_trp": {2020: 1500, 2030: 900, 2050: 300, 2100: 250}, "h2_trp": {2020: 2500, 2050: 800, 2100: 600},
}

# Activity in 2020 (GWa/yr): approximate, rounded values. Total electricity generation
# is of the order reported by the IEA for 2020 (about 26,000-27,000 TWh, of which
# 7,593 TWh renewable); total final energy of the order of 385-400 EJ.
WORLD_HIST_2020 = {
    "coal_ppl": 1070, "gas_ppl": 720, "nuc_ppl": 310, "hydro_ppl": 490, "wind_ppl": 180,
    "solar_pv": 95, "bio_ppl": 75, "elec_t_d": 2650, "elec_spec": 2610,
    "coal_i": 1000, "gas_i": 1615, "oil_i": 1145, "bio_i": 635,
    "oil_trp": 755, "gas_trp": 30, "bio_trp": 20, "elec_trp": 30,
}

# Baseline demand for useful energy (GWa/yr) in 2020 and annual growth rates
# (to 2050, 2050-2070, after 2070): an SSP2-like "middle of the road" assumption.
WORLD_DEMAND_2020 = {"spec": 2610, "therm": 4395, "transport": 840}
WORLD_DEMAND_GROWTH = {"spec": (0.025, 0.015, 0.010), "therm": (0.010, 0.005, 0.003),
                       "transport": (0.020, 0.010, 0.007)}

# Dynamic constraints: maximum annual growth rate and start-up value (GWa/yr)
WORLD_GROWTH = {
    "solar_pv": (0.10, 30), "wind_ppl": (0.10, 30), "nuc_ppl": (0.05, 5), "hydro_ppl": (0.03, 5),
    "coal_ccs": (0.15, 10), "gas_ccs": (0.15, 10), "bio_ccs": (0.15, 10), "bio_ppl": (0.08, 5),
    "h2_elec": (0.20, 10), "h2_i": (0.15, 10), "h2_trp": (0.15, 5),
    "elec_trp": (0.15, 10), "elec_i": (0.10, 20), "gas_ppl": (0.06, 20), "coal_ppl": (0.05, 10),
    "coal_i": (0.02, 10), "gas_i": (0.04, 20), "oil_i": (0.02, 10), "bio_i": (0.02, 10),
    "oil_trp": (0.03, 10), "gas_trp": (0.05, 5), "bio_trp": (0.08, 5),
}

# Maximum annual rate of decline of existing technologies (growth_activity_lo). Capital
# stock, infrastructure and habits change slowly, which an LP ignores unless told.
WORLD_DECLINE = {**{t: -0.06 for t in ("coal_ppl", "gas_ppl")},
                 **{t: -0.04 for t in ("coal_i", "gas_i", "oil_i", "bio_i", "oil_trp", "gas_trp")}}

# Resource and potential limits (GWa/yr)
WORLD_BOUNDS = {"bio_extr": 3800,   # about 120 EJ/yr of biomass
                "hydro_ppl": 900}   # about 7,900 TWh/yr of hydropower

VRE_SHARE = 0.6  # maximum share of wind and solar in electricity generation


def _interp(points, year):
    xs = sorted(points)
    return float(np.interp(year, xs, [points[x] for x in xs]))


def _growth_path(v0, rates, years, y0=2020):
    out, v, prev = {}, v0, y0
    for y in years:
        for yy in range(prev + 1, y + 1):
            v *= 1 + (rates[0] if yy <= 2050 else rates[1] if yy <= 2070 else rates[2])
        out[y], prev = v, y
    return out


def build_world(scen, node="World"):
    """Populate ``scen`` (a miniiam or message_ix Scenario) with the global dataset."""
    _register_units(scen)
    years = WORLD_YEARS
    horizon = [y for y in years if y >= WORLD_FIRST_YEAR]
    scen.add_horizon(year=years, firstmodelyear=WORLD_FIRST_YEAR)
    _add_node(scen, node)
    scen.add_set("mode", "standard")
    scen.add_set("technology", list(WORLD_TECH))
    scen.add_set("commodity", sorted({v[2] for v in WORLD_TECH.values()} | {"crudeoil"}))
    scen.add_set("level", ["primary", "secondary", "final", "useful"])
    scen.add_set("emission", "CO2")
    scen.add_cat("emission", "GHG", "CO2")
    scen.add_par("interestrate", make_df("interestrate", year=horizon, value=0.05, unit="-"))

    va = {t: _va_pairs(years, horizon, WORLD_CAPACITY.get(t, {}).get("life")) for t in WORLD_TECH}
    for t, (ic, il, oc, ol, eff) in WORLD_TECH.items():
        _add_tec(scen, node, t, va[t], (oc, ol, 1.0),
                 None if ic is None else (ic, il, round(1 / eff, 6)))
        if t in WORLD_CAPACITY:
            vc = [WORLD_CAPACITY[t]["var"]] * len(va[t])
        else:
            vc = [_interp(WORLD_VAR_COST.get(t, {2020: 0}), a) for _, a in va[t]]
        scen.add_par("var_cost", make_df("var_cost", node_loc=node, technology=t,
                                         year_vtg=[p[0] for p in va[t]], year_act=[p[1] for p in va[t]],
                                         mode="standard", time="year", value=vc, unit="USD/kWa"))

    for t, d in WORLD_CAPACITY.items():
        yv, ya = [p[0] for p in va[t]], [p[1] for p in va[t]]
        scen.add_par("inv_cost", make_df("inv_cost", node_loc=node, technology=t, year_vtg=horizon,
                                         value=[_interp(d["inv"], y) for y in horizon], unit="USD/kW"))
        scen.add_par("technical_lifetime", make_df("technical_lifetime", node_loc=node, technology=t,
                                                   year_vtg=years, value=d["life"], unit="y"))
        scen.add_par("fix_cost", make_df("fix_cost", node_loc=node, technology=t, year_vtg=yv,
                                         year_act=ya, value=d["fix"], unit="USD/kWa"))
        scen.add_par("capacity_factor", make_df("capacity_factor", node_loc=node, technology=t,
                                                year_vtg=yv, year_act=ya, time="year",
                                                value=d["cf"], unit="-"))
        # Historical capacity: enough to deliver the 2020 activity, spread over past
        # vintages still alive in 2020 (only recent vintages for wind and solar)
        act0 = WORLD_HIST_2020.get(t, 0)
        if act0 > 0:
            past = [v for v in years if v <= 2020 and 2020 - v < d["life"]]
            shares = {2010: 0.3, 2020: 0.7} if t in ("wind_ppl", "solar_pv") else {v: 1 / len(past) for v in past}
            scen.add_par("historical_new_capacity", make_df(
                "historical_new_capacity", node_loc=node, technology=t, year_vtg=list(shares),
                value=[act0 / d["cf"] * s_ / 10 for s_ in shares.values()], unit="GW/y"))

    scen.add_par("historical_activity", make_df(
        "historical_activity", node_loc=node, technology=list(WORLD_HIST_2020), year_act=2020,
        mode="standard", time="year", value=list(WORLD_HIST_2020.values()), unit="GWa"))

    for sec, v0 in WORLD_DEMAND_2020.items():
        path = _growth_path(v0, WORLD_DEMAND_GROWTH[sec], horizon)
        scen.add_par("demand", make_df("demand", node=node, commodity=sec, level="useful",
                                       year=horizon, time="year",
                                       value=[round(path[y], 1) for y in horizon], unit="GWa"))

    for t, (g, start) in WORLD_GROWTH.items():
        scen.add_par("growth_activity_up", make_df("growth_activity_up", node_loc=node, technology=t,
                                                   year_act=horizon, time="year", value=g, unit="-"))
        scen.add_par("initial_activity_up", make_df("initial_activity_up", node_loc=node, technology=t,
                                                    year_act=horizon, time="year", value=start, unit="GWa"))
    for t, g in WORLD_DECLINE.items():
        scen.add_par("growth_activity_lo", make_df("growth_activity_lo", node_loc=node, technology=t,
                                                   year_act=horizon, time="year", value=g, unit="-"))
    for t, ub in WORLD_BOUNDS.items():
        scen.add_par("bound_activity_up", make_df("bound_activity_up", node_loc=node, technology=t,
                                                  year_act=horizon, mode="standard", time="year",
                                                  value=ub, unit="GWa"))

    # CO2 is counted when fossil fuels are extracted; CCS plants capture part of the CO2
    # contained in their fuel (for bio_ccs this gives net negative emissions)
    ef_rows = {"coal_extr": EF["coal"], "gas_extr": EF["gas"], "oil_extr": EF["oil"],
               "coal_ccs": -CAPTURE * EF["coal"] / 0.30, "gas_ccs": -CAPTURE * EF["gas"] / 0.45,
               "bio_ccs": -CAPTURE * EF["biomass"] / 0.25}
    for t, val in ef_rows.items():
        scen.add_par("emission_factor", make_df(
            "emission_factor", node_loc=node, technology=t, year_vtg=[p[0] for p in va[t]],
            year_act=[p[1] for p in va[t]], mode="standard", emission="CO2",
            value=round(val, 4), unit="tCO2/kWa"))

    # Relation "vre_share": wind + solar <= VRE_SHARE x total generation, written as
    # (1 - share) x ACT(wind, solar) - share x ACT(other plants) <= 0
    scen.add_set("relation", "vre_share")
    gen = [t for t, v in WORLD_TECH.items() if v[2] == "electricity" and v[3] == "secondary"]
    for t in gen:
        coef = (1 - VRE_SHARE) if t in ("wind_ppl", "solar_pv") else -VRE_SHARE
        scen.add_par("relation_activity", make_df(
            "relation_activity", relation="vre_share", node_rel=node, year_rel=horizon,
            node_loc=node, technology=t, year_act=horizon, mode="standard", value=coef, unit="-"))
    scen.add_par("relation_upper", make_df("relation_upper", relation="vre_share", node_rel=node,
                                           year_rel=horizon, value=0.0, unit="-"))
    return scen


def world_macro_data(node="World"):
    """MACRO input data for the World model, in the format of message_ix add_macro().

    GDP is in billion USD (2020 prices, market exchange rates), the unit that
    MESSAGE-MACRO expects: about 85,000 (85 trillion) in 2020, with SSP2-like growth. esub, kpvs, kgdp, depr, drate and lotol take the values
    used in the message_ix test data. price_ref and cost_ref are omitted, so they are
    extrapolated from the baseline solution, as message_ix does.
    """
    horizon = [y for y in WORLD_YEARS if y >= WORLD_FIRST_YEAR]
    gdp = _growth_path(85.0, (0.028, 0.020, 0.015), horizon)
    gdp.update({2010: 67.0, 2020: 85.0})
    gdp_years = [2010, 2020] + horizon
    sectors = list(WORLD_DEMAND_2020)
    one = lambda v: pd.DataFrame({"node": [node], "value": [v], "unit": ["-"]})
    return {
        "config": pd.DataFrame({"node": [node, None, None], "sector": sectors,
                                "commodity": sectors, "level": ["useful"] * 3, "year": horizon[:3]}),
        "gdp_calibrate": pd.DataFrame({"node": node, "year": gdp_years,
                                       "value": [round(1e3 * gdp[y], 0) for y in gdp_years], "unit": "G$"}),
        "demand_ref": pd.DataFrame({"node": node, "sector": sectors,
                                    "value": [WORLD_DEMAND_2020[s] for s in sectors], "unit": "GWa"}),
        "esub": one(0.3), "drate": one(0.05), "depr": one(0.05), "kpvs": one(0.3),
        "kgdp": one(3.0), "lotol": one(0.05),
        "aeei": pd.DataFrame([(node, s, y, 0.02, "-") for s in sectors for y in horizon],
                             columns=["node", "sector", "year", "value", "unit"]),
    }
