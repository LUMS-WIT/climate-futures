# miniam

A zero-install Python port of a reference GAMS integrated-assessment course
model (linopy + HiGHS), built to a MESSAGE-like equation skeleton so it reads
next to the real MESSAGEix documentation.

## Setup

```
pip install -r requirements.txt
```

Nothing beyond that — no GAMS, no compiler, no system library.

## Layout

```
miniam/
  core/formulation.py   the LP itself — GAMS equation names preserved as constraint names
  data/                 world_model.yaml, simple_model.yaml, loader.py
  climate/fair_link.py  soft link to the fair package — interface only, not implemented
  macro/macro_link.py   iterative link to a CES demand model — interface only, not implemented
  viz/                  not started
  run/energy.py         driver: build, solve, report
tests/
```

## Running it

```
python -m miniam.run.energy --dataset world
python -m miniam.run.energy --dataset simple
python -m miniam.run.energy --dataset world --cum-emiss-up 500
```

```
pytest tests/
```

`test_simple_model.py` checks the toy model (share constraint binds at 0.4).
`test_world_model.py` checks the full model (commodity balance closes at
every level and period; a tight cumulative CO2 cap reduces coal activity and
binds exactly at the cap) — structural checks, not pinned magnitudes, since
no GAMS reference run exists for this port to pin them against.

## Editing the data

`world_model.yaml` / `simple_model.yaml` hold every parameter — technology
list, cost tables, lifetimes, demand, share constraints. Add a technology or
change a cost by editing the YAML; `loader.py` is the only code that needs to
understand the file, and it only computes the two things that genuinely need
computing (the capital-recovery-factor annuity behind `cost_capacity`, and
GDP-scaled demand growth) rather than storing pre-computed numbers.

## Splitting the world model into two regions

The core as it stands is single-region (global) — there is no `region`
dimension yet anywhere in `formulation.py` or the YAML schema. This is the
concrete starting checklist for the smallest version of that change: two
regions, not the full R5/R10 generalization it would eventually grow into.

1. **Add `region` to `EnergySystemData` and thread it through
   `formulation.py`.** `ACT`, `CAP_NEW`, and `EMISS` all gain a `region`
   coordinate dimension alongside `technology`/`year`. Region-invariant
   parameters (technology techno-economics — `inv`, `fom`, `vom`, `lifetime`,
   `hours`, `co2_emission`, `diffusion_up`, `startup`) can stay shared across
   regions unless a specific exercise wants them to vary too; region-specific
   parameters (`demand`, `act_fx`, `act_lo`) need the extra key.
2. **Split `EQ_ENERGY_BALANCE` per region.** A technology's output only
   satisfies demand *in its own region*, with one exception below. This is
   the main code change: the constraint-building loop currently groups by
   `(energy, level)` only (see `formulation.py`'s
   `EQ_ENERGY_BALANCE` block) — it needs to group by
   `(region, energy, level)` instead.
3. **Model trade as technologies, not as a global pool.** For each tradeable
   commodity, add a technology whose input region differs from its output
   region — e.g. `coal_trade_A_to_B` consumes `coal.final` in region A and
   produces it in region B. This is what MESSAGE does. The alternative —
   `EXPORT`/`IMPORT` variables against a global pool — is easier to code but
   teaches a market structure no real IAM uses.
4. **Give every trade technology a small nonzero `vom`.** With identical
   techno-economics in both regions and a *free* trade technology, the
   solver has no reason to prefer one trade pattern over another — different
   solver versions will report different (equally optimal) flows, which
   reads as a bug rather than what it is (degeneracy). A small transport
   cost fixes this and is worth adding before running the model even once,
   not after debugging a "nondeterministic" result.
5. **Start with the simplest possible split.** Keep technology
   techno-economics identical in both regions; only `demand` differs, split
   by a simple share (e.g. population or GDP share, even a flat 60/40 to
   start). This isolates what a two-region model actually changes — trade
   flows and the resulting cost/technology-mix shift — rather than
   entangling it with also-recalibrating every regional cost.
6. **Keep the aggregation in YAML, not code**, per the pattern `loader.py`
   already uses for everything else: a `regions` list in the YAML (even a
   two-entry one) rather than a hardcoded region count anywhere in
   `formulation.py`. This is what makes a later R5/R10 generalization a data
   change, not a rewrite.
