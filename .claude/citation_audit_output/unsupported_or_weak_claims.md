# Sentences Requiring Attention

Sentences grouped by failure mode. Items at the top are the most impactful for thesis defensibility.

## Contradicted (highest priority)

### TBL-Li — Li 2024 table row attributes a simulation number to "prototype tests"
- **Original**: "Li et al. 2024 — PCC head-following motion planner (**prototype tests**) — Avg. path-tracking error — 7.89 mm"
- **What Li actually says**: 7.8928 mm is the **average tracking accuracy across three simulated paths**, comparing the PAHF algorithm to PSO, FABRIK, and CRRIK in simulation. Section 3.2, Fig. 8 caption says "Path-tracking simulation results." Prototype results in §4.2 are reported only qualitatively, with no 7.89 mm number attached.
- **Severity**: Critical for table accuracy. A reader who checks Li 2024 will see the number is simulation-only.

## Overstated / Partial support that the prose implies as Strong

### S28 — "Learning-based models can capture nonlinear and history-dependent behavior, including hysteresis"
- **Issue**: Wolfe 2024 is cited but Wolfe's ANN is **purely kinematic** — it does not capture hysteresis. Wolfe identifies hysteresis statistically (L3962–3970), simulates it on the Camarillo analytical model, and proposes a history-dependent learned model f̂(u(k), u(k−1)) as a **future direction** (L3970–3980, L4015–4021, L4167–4168). Rao acknowledges learned models exist but explicitly excludes them from review scope.
- **Severity**: Material — the claim implies Wolfe demonstrated hysteresis-capture, which Wolfe did not.

### S26 — Cosserat / FE "at greater computational and parameter-identification cost"
- **Issue**: Rao addresses Cosserat (variable-curvature) vs PCC, not finite-element models. Oliver-Butler actually argues their Cosserat implementation CAN run in real time (L1257–1260), undercutting the "greater computational cost" framing. "FE" specifically is not represented in the cited sources.
- **Severity**: Moderate.

### S27 / TBL-Raimondi-PCC — "4.10%"
- **Issue**: Source says "4.1%" (one decimal). Reporting "4.10%" implies more precision than Raimondi reports.
- **Severity**: Minor but easy to fix.

## Weakly supported / mismatched citations

### S15 (C3) — "tendon actuation is extrinsic — actuators can be grounded off the robot"
- **Issue**: Rao verbatim supports this (L68–69). Camarillo does not use "extrinsic" framing — it is implicit at best.
- **Recommended action**: Cite Rao alone for the extrinsic framing.

### S19 (C6) — "Tendon state, pretension, routing geometry, friction, and backbone compliance"
- **Issue**: **"Pretension"** is not named as a repeatability factor by any of the four cited sources (Camarillo, Rao, Raimondi, Shihora).
- **Recommended action**: Drop "pretension" from the list OR add Wolfe (who discusses pretension and tendon-stretch effects on loading-state shape) as a citation here.

### S24 (C10) — "friction, tendon stretch, actuator compliance, material hysteresis, backbone compression, and routing losses"
- **Issue**: Multiple sub-issues:
  - **"Routing losses"** has no clear anchor in any of the five cited sources (closest is friction along path)
  - **Rao does NOT cover** backbone compression or material hysteresis
  - Actuator compliance is only indirectly mentioned (Camarillo)
- **Recommended action**: Drop "routing losses" or replace with "tendon-path friction." Split into per-factor sub-citations rather than bundling.

### S30 — Platform-induced model error list
- **Issue**: "slack, inconsistent pretension, changing tendon seating, unvalidated tracking, or undocumented calibration state" is the author's synthesis, not directly anchored in any cited source. Shihora does not discuss learning or datasets at all (purely a design-stage mechanics paper).
- **Recommended action**: Soften the list (e.g., "such as unmeasured slack or undocumented calibration") or move Shihora citation to S19 where it fits better.

### S34 — Four-modality sensing list
- **Issue**: No single cited survey enumerates all four modalities. Shi covers FBG, EM, and intraoperative imaging. Sincak covers electricity, magnetism, optics. The four-item list is the UNION.
- **Recommended action**: Cite Shi for "intraoperative imaging-based reconstruction" specifically; cite Sincak for "flexible electric or magnetic sensors embedded in the robot body." OR rephrase as "across recent surveys."

### S36 — "robot frame"
- **Issue**: He frames the transform as field-generator-to-sensor; Shi frames registration as to preoperative imaging. Neither uses "robot frame." This is author synthesis.
- **Recommended action**: Acceptable, but acknowledge "an application-specific frame" or note the synthesis.

### S39 — Six-factor "experimental behavior" list
- **Issue**: Grassmann is cited but only supports fabrication tolerances + friction + actuation hardware. Routing details, material response, and sensing integration are NOT enumerated in Grassmann.
- **Recommended action**: Restrict Grassmann citation to the elements it actually covers, OR drop Grassmann from this sentence and rely on Rao+Raimondi.

### S42 — Seven-element protocol checklist
- **Issue**: Grassmann argues for reproducibility/verification/benchmarking beyond mere sharing, but does NOT itemize "assembly, initialization, pretensioning, tracking, registration, commanding, and logging."
- **Recommended action**: Frame the checklist as the thesis's own synthesis motivated by Grassmann's call for reproducibility infrastructure.

## Partial — minor (acceptable as written, noted for awareness)

### S3 — "improved visualization, motion scaling, and distal dexterity"
- "Motion scaling" not in those exact words in Dupont; concept covered.

### S7 — "distribute contact forces over a compliant body"
- "Distribute contact forces" is author paraphrase of "gently conform" / "inherent safety." Conceptually equivalent.

### S29 / TBL-Wolfe — "2.2 mm" and "9.5 mm"
- Abstract values; body text uses 2.24 / 9.50 mm. Either is defensible; body values are more precise.

## Sentences with no citation that arguably need one (no failures, opportunities)

- **S5** ("Continuum robots have emerged as one response to this need."): topic sentence; could cite Burgner-Kahrs or Russo
- **S17** ("The architecture is mechanically intuitive, but it is not equivalent to a rigid-link robot…"): could cite Webster L105–108
- **S20** ("The same compliance that makes continuum robots clinically appealing makes them difficult to model and control."): topic sentence; could cite Burgner-Kahrs + Webster
- **S33** ("The pose of the tip depends on the full mechanical state of the robot…"): could cite Camarillo (hysteresis) or Oliver-Butler (loading-state dependence)
- **S38** ("Recent continuum robotics research has emphasized the need for better platforms…"): could be supported by Grassmann L83–92

## Source unreadable / unavailable

- None. All 20 .txt files in `/Users/jacklange/Downloads/txtResearch/` were readable.
- **Note**: `Continuum_Robot_Stiffness_Under_External_Loads_and_Prescribed_Tendon_Displacements.txt` is a byte-identical duplicate of `Oliver-Butler et al. - 2019 - …` (same size 77821 bytes, same 1717 lines). Treated as one source.
- **Note**: `Sozer et al. - 2023 - Robotic Modules for a Continuum Manipulator With Variable Stiffness Joints.txt` is present in the folder but is **not cited anywhere in the Background section**. Not audited.

## Most important fixes before submission (ranked)

1. **Fix TBL-Li**: "prototype tests" → "simulation, avg across three paths." This is the only outright contradicted item.
2. **Soften S28 (Wolfe + learned hysteresis)**: Wolfe's ANN does not capture hysteresis; rewrite to reflect "future direction" framing OR add a citation that actually demonstrates a hysteresis-aware learned model.
3. **Fix S27 / TBL-Raimondi**: "4.10%" → "4.1%" (matches source precision).
4. **Drop "pretension" from S19** OR add a citation that names it (Wolfe is closest).
5. **Drop "routing losses" from S24** OR change to "tendon-path friction"; reconsider Rao for backbone compression / material hysteresis.
6. **Restrict Grassmann citation in S39** to fabrication-tolerance + actuation portion only.
7. **Split S34 citation** so each modality maps to a survey that actually enumerates it.
8. **Demote Camarillo in S15 (C3)** — Rao is the right cite for "extrinsic actuation."
9. **Reframe S42's seven-element protocol** as the thesis's own synthesis, motivated by Grassmann.
