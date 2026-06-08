# Research Directions — Where This Can Become a Paper

An honest assessment of what in this codebase is genuinely novel vs. well-trodden,
and the concrete experiment that would turn each novel angle into a publishable
result. Written for the next researcher/student.

**The blunt framing first.** Most individual pieces here are standard practice and
will not publish on their own: SVD/Procrustes + RANSAC registration, RMS-to-centroid
tip repeatability (ISO 9283 style), a cable→tip MLP benchmarked against Wolfe,
and "use servo present-current as a cheap force proxy." All of these are competent
engineering and necessary infrastructure, but each is well-covered in the
continuum/soft-robotics literature. The novelty is in three *combinations and
framings* that, as far as the literature search goes, are not packaged anywhere
else. They are ranked below by how defensible the novelty is and how close the
infrastructure already is to producing the result.

> **The one caveat that gates everything:** per `README.md` → *Current Project
> Status*, this is largely **validated infrastructure with little-to-no live bench
> data**. Every direction below is real and the code to run it exists — but the
> publishable result requires actually collecting the hardware dataset. The
> contribution is "we built the protocol and here is the data it produced," not the
> protocol description alone.

---

## Direction 1 — A load-cell-free, repeatable pretensioning protocol (strongest)

**The claim.** Tendon-driven continuum robots are notoriously sensitive to their
*startup tension state*, yet most work either instruments every tendon with a load
cell (expensive, bulky, rarely deployed) or sets a fixed pre-tension once and
ignores run-to-run repeatability. This system proposes a **load-cell-free
pretensioning protocol** that is repeatable enough to gate downstream experiments,
using only the servo's own signed current.

**What is actually novel (vs. "current as a force proxy," which is not):**

1. **Holding-current vs. motion-current separation.** The key insight — and a
   documented failure that motivated it — is that an instantaneous current read
   taken right after a goal-position write reports *drive torque* (e.g. `+17 mA`
   while the motor is still closing on its setpoint), not the steady-state *holding*
   current at that position (`-30 mA`) — a ~45 mA error that silently breaks any
   tension decision. The protocol takes a **settled burst average** (sleep
   `post_move_settle_s`, then average `holding_current_burst_count` reads) and uses
   the per-sample standard deviation as a "motion hasn't decayed yet" guard. I have
   not seen this measurement protocol formalized for tendon robots.
2. **Tighten-only tension equalization as a startup primitive.** After take-up, the
   protocol drives all tendons to within a few mA of each other by *only tightening
   the low ones, never releasing* — justified by the claim that **spine friction
   dominates small tip motions, so balanced tendon force is a more repeatable
   startup invariant than tip position**. That is a falsifiable, interesting
   hypothesis and the inverse of what a tip-centering controller would do.
3. **A formal repeatability verdict** ("one-rig proof": 5 repeats graded
   high/medium/low/failed on accepted-fraction + tip scatter + per-servo tension std).

**Why it's publishable:** it reframes pretensioning from "a thing you do once" to
"a measured, gated, repeatable startup state with an acceptance test" — a methods
contribution for low-cost tendon robotics.

**The killer experiment (what's missing):**
- **Ground-truth one tendon against an actual load cell** across the −5 to −50 mA
  range. Establish the mA→Newton map, its hysteresis, and its repeatability. This
  converts the operator-calibrated "−30 mA = tight" scale into a defensible
  calibrated claim.
- Run the 5-repeat verdict **across power-cycles and re-mounts** (not just back-to-
  back), and report the distribution of (a) final tendon-tension spread and (b)
  resulting tip scatter. The headline figure is "tip repeatability vs. pretension
  balance" — proving the friction/balance hypothesis with data.
- Ablation: protocol with vs. without the holding-current settle, with vs. without
  tighten-only equalization. Show each component reduces tip scatter.

**Venue:** RA-L / ICRA / RoboSoft (methods + hardware validation).

---

## Direction 2 — Trust-gated experimental data collection as reproducibility methodology

**The claim.** Hardware-robotics results are hard to reproduce partly because the
*provenance and trustworthiness of the collected data* is implicit — a dataset
collected from an unverified calibration/startup state can silently contaminate a
model or a precision claim. This codebase implements a **machine-checked,
capture-time trust gate**: every run records `run_trust_mode`,
`valid_for_thesis_repeatability`, `valid_for_model_training`, and
`data_quality_warnings`; the runtime-tip policy propagates trust into every sample;
and the evidence-index builder *refuses to let a run's curation label override its
measured validity* (an operator-tagged "thesis candidate" is still downgraded to
`lower_trust` with a stated reason if its own summary contradicts the label).

**Why it's novel:** trust/provenance metadata exists in many systems, but a
*named, automatic, non-overridable, capture-to-analysis* trust gate for
continuum/soft-robotics experiments — where the gate is computed from measured
state (registration FRE, tracker freshness, pretension acceptance, tip-pose trust)
and cannot be defeated by reviewer optimism — is, as far as I can tell, not
packaged in the literature. It speaks directly to the robotics reproducibility
conversation.

**The killer experiment (what's missing):**
- **Show the gate changes the conclusion, not just the bookkeeping.** Collect two
  datasets — one gated, one ungated (including runs the gate would reject) — and
  show they produce *materially different* model RMSE or repeatability numbers. If
  gated-vs-ungated differs by, say, 30%, that is the whole paper: "un-gated hardware
  data overstates precision by X%."
- The codebase already has documented near-misses to use as case studies: the
  `0/+17 mA` loose-pretension run, the asymmetric-baseline false-accept (a 5 mA real
  tendon-spread that looked like 20 mA under the legacy `|current − baseline|`
  metric), and a radial-vs-planar std mislabel caught by audit.

**Venue:** a reproducibility / datasets / benchmarking track (e.g. ICRA repro
workshop, or a journal methods paper). This is the most *transferable* contribution —
it isn't specific to continuum robots.

---

## Direction 3 — Longitudinal tendon/neutral drift characterization

**The claim.** Tendon creep, spool slippage, and knot settling cause the mechanical
*zero* of a tendon-driven robot to drift over its operational life. This is widely
acknowledged as a nuisance and almost never *quantified longitudinally*. This
codebase now **logs every neutral/zero/pretension update to an append-only JSONL
audit trail** (`neutral_zero_log.jsonl`) and ships an analysis experiment that
builds per-servo drift timelines (`neutral_drift_ticks`, `max_excursion_from_first`,
per-field std) with thesis figures.

**Why it's novel:** the *instrumented, automatic, event-tied longitudinal drift
log* is the under-explored piece. Most papers re-calibrate and move on; a clean
"how much does the zero of a tendon robot drift over N operating hours/cycles, is it
monotonic creep or random walk, and what drives it" characterization — framed as a
**recalibration-scheduling / maintenance** argument — is a legitimate small
contribution.

**The killer experiment (what's missing):**
- Accumulate **weeks/months of real bench data** in the log.
- Correlate `neutral_drift_ticks` against cycle count, cumulative tendon-tension
  history, and power-cycle count. Classify the drift (creep vs. random walk).
- Close the loop with Direction 1: **show that un-tracked neutral drift degrades
  repeatability and that periodic re-zeroing recovers it.** That makes the logging
  actionable rather than descriptive.

**Venue:** a focused hardware/characterization paper or a strong workshop; or fold
it into Direction 1 as the "why startup state must be re-measured" section.

---

## The paper I would actually write

Combine 1 + 3 into a single hardware-methods paper and use 2 as the rigor backbone:

> **"Repeatable, load-cell-free startup-state control for tendon-driven continuum
> robots, and its effect on positioning repeatability over the robot's operational
> life."**
>
> 1. The pretensioning protocol (holding-current measurement + tighten-only tension
>    equalization) — validated against a single load cell.
> 2. A repeatability study showing tip scatter as a function of tendon-tension
>    balance (the friction hypothesis), across power-cycles.
> 3. A longitudinal drift section: neutral drift over time, its effect on
>    repeatability, and a re-zeroing schedule that recovers it.
> 4. All of it collected through the trust-gated pipeline (Direction 2), so the
>    numbers are defensible — with one concrete "the gate caught a bad run" case study.

This is achievable with the existing code; it needs **(a) one load cell, (b) a few
weeks of disciplined bench data collection, and (c) the gated-vs-ungated
comparison.** No new infrastructure required — the experiments, logging, trust
gates, and figure pipelines are already built and tested.

## What NOT to try to publish alone
- Registration accuracy (sub-mm FRE with Aurora EM is expected, textbook solver).
- Tip repeatability numbers in isolation (standard metric; only interesting when
  *paired with a cause* — tension balance or drift).
- The cable→tip MLP (explicitly benchmarked against Wolfe; well-trodden). The
  `HybridResidualModel` (physics + learned residual) could be interesting *only*
  with a strong analytic baseline and a clear, characterized win over pure-ANN.

---

*Key code references: `continuum_robot/experiments/builtins.py` (pretension
pipeline, consistency verdict), `config/experiment_pretension_validation.example.yaml`,
`continuum_robot/experiments/single_segment_repeatability.py`,
`continuum_robot/experiments/neutral_setpoint_drift_validation.py`,
`continuum_robot/servos/neutral_calibration_service.py`,
`continuum_robot/tracking/runtime_tip_policy.py`,
`continuum_robot/data/build_thesis_evidence_index.py`,
`continuum_robot/modeling/ann_training.py`.*
