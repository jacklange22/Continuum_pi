# Suggested Revisions

For each problematic sentence: original, issue, suggested revision, and rationale. Ordered by severity.

---

## 1. TBL-Li (Critical — contradicted by source)

**Original (table row)**
```
Li et al. 2024 — PCC head-following motion planner (prototype tests) —
Avg. path-tracking error — 7.89 mm
```

**Issue**: The 7.8928 mm figure is the average across three **simulated** path-tracking trajectories (Li §3.2, Fig. 8 captioned "Path-tracking simulation results"). Li's prototype experiments in §4.2 report no comparable number; they describe accuracy only qualitatively ("exceptional tracking accuracy").

**Suggested revision (table row)**
```latex
Li et al.\ 2024~\cite{li_flexible_2024} &
PCC head-following motion planner (simulation, avg across three paths) &
Avg.\ path-tracking error (simulation) &
$7.89$~mm \\
```

**Rationale**: Reflects the actual experimental condition. A reader who checks Li will see the number and the context match. If you want a prototype number, Li does not provide one and this row would need to be removed or replaced.

---

## 2. S28 — Learned models and hysteresis (Overstated)

**Original**
> Learning-based models can capture nonlinear and history-dependent behavior, including hysteresis, without explicitly modeling every physical mechanism~\cite{rao_how_2021,wolfe_learned_2024}.

**Issue**: Wolfe's actual learned model is `f̂(u)` — purely kinematic, with no time-history input. Wolfe identifies hysteresis empirically (1.181 mm cluster RMS, ANOVA p = 4.85×10⁻⁷ at L3962–3970), simulates hysteresis on the Camarillo *analytical* model (L3913–3946), and explicitly proposes a hysteresis-aware learned model `f̂(u(k), u(k−1))` as **future work** (L3970–3980, L4015–4021, L4167–4168). Rao only acknowledges that data-driven approaches exist (L107–110), without endorsing the hysteresis-capture capability.

**Suggested revision**
> Learning-based models offer a route to capturing nonlinear and history-dependent behavior such as hysteresis, in principle without explicitly modeling every physical mechanism, although the historical Thayer ANN forward model is kinematic and the use of past inputs to capture hysteresis remains an identified future direction~\cite{rao_how_2021,wolfe_learned_2024}.

**Rationale**: Honest about Wolfe's actual demonstration vs. future-work framing, while keeping the conceptual claim about learning's potential.

---

## 3. S27 / TBL-Raimondi — "4.10%" → "4.1%" (Precision)

**Original (prose)**
> ...compared with $4.10\%$ and $13.86\%$ for a PCC kinematic baseline on the same robot~\cite{raimondi_understanding_2024}.

**Suggested revision**
> ...compared with $4.1\%$ and $13.86\%$ for a PCC kinematic baseline on the same robot~\cite{raimondi_understanding_2024}.

**Rationale**: Raimondi consistently writes "4.1%" with one decimal place (abstract L36, conclusion L758, L832). Writing 4.10% over-reports the precision the source actually provides. Same fix for the table row.

---

## 4. S19 (C6) — "pretension" not anchored

**Original**
> Tendon state, pretension, routing geometry, friction, and backbone compliance therefore all affect whether a commanded tendon displacement produces the same robot shape on repeated trials~\cite{camarillo_mechanics_2008,rao_how_2021,raimondi_understanding_2024,shihora_friction-limited_2024}.

**Issue**: None of the four cited sources names "pretension" as a repeatability factor. Camarillo and Rao discuss tendon stretch and slack; Raimondi mentions pretension only as a setup condition; Shihora analyses friction-induced resolution. The phrase as a *factor list* is the thesis's own; pretension as a *factor* is not in the cited papers.

**Suggested revision (option A — drop pretension)**
> Tendon state, routing geometry, friction, and backbone compliance therefore all affect whether a commanded tendon displacement produces the same robot shape on repeated trials~\cite{camarillo_mechanics_2008,rao_how_2021,raimondi_understanding_2024,shihora_friction-limited_2024}.

**Suggested revision (option B — keep pretension, add appropriate cite)**
> Tendon state, pretension, routing geometry, friction, and backbone compliance therefore all affect whether a commanded tendon displacement produces the same robot shape on repeated trials~\cite{camarillo_mechanics_2008,rao_how_2021,oliver-butler_continuum_2019,raimondi_understanding_2024,shihora_friction-limited_2024}.

(Oliver-Butler L1224–1240 demonstrates that tendon stretch + control of proximal tendon ends during loading affects loaded shape, which is the closest in-folder support for a pretension-style factor.)

**Rationale**: Option A is the cleanest. Option B keeps the bullet but adds Oliver-Butler whose discussion of "how proximal tendon ends are controlled during loading" is the nearest anchor for pretension.

---

## 5. S24 (C10) — "routing losses", Rao on backbone compression / hysteresis

**Original**
> Physical tendon-driven robots, however, rarely satisfy the underlying assumptions: friction, tendon stretch, actuator compliance, material hysteresis, backbone compression, and routing losses cause the same commanded input to produce different measured shapes depending on prior motion and loading state~\cite{camarillo_mechanics_2008,rao_how_2021,oliver-butler_continuum_2019,raimondi_understanding_2024,shihora_friction-limited_2024}.

**Issue**: (a) "Routing losses" has no anchor in any cited source. (b) Rao does not discuss backbone compression or material hysteresis. (c) Actuator compliance is only indirectly addressed (Camarillo).

**Suggested revision**
> Physical tendon-driven robots, however, rarely satisfy the underlying assumptions: friction, tendon stretch, material hysteresis, and backbone compression cause the same commanded input to produce different measured shapes depending on prior motion and loading state~\cite{camarillo_mechanics_2008,oliver-butler_continuum_2019,raimondi_understanding_2024,shihora_friction-limited_2024}.

**Rationale**: Drops "actuator compliance" and "routing losses" (which are not well anchored), and drops Rao from the citation list because Rao does not cover the remaining specific items. Keeps Camarillo (hysteresis + backbone compression + stretch), Oliver-Butler (stretch, compression, loaded shape), Raimondi (friction, superelasticity), Shihora (friction, wire compliance).

---

## 6. S30 — Platform-induced learned-model error list

**Original**
> Learned models are, however, only as trustworthy as the datasets used to train and evaluate them: if repeated trials differ because of unmeasured slack, inconsistent pretension, changing tendon seating, unvalidated tracking, or undocumented calibration state, then model error reflects the experimental platform as much as the underlying robot~\cite{rao_how_2021,camarillo_mechanics_2008,raimondi_understanding_2024,shihora_friction-limited_2024}.

**Issue**: The five-element platform-factor list is the thesis's own synthesis. None of the cited sources enumerates "slack, pretension, seating, tracking, calibration" as such. Shihora doesn't discuss learning at all. Rao supports only the general principle (L2060–2113: model error reflects calibration choices).

**Suggested revision**
> Learned models are, however, only as trustworthy as the datasets used to train and evaluate them. If repeated trials differ because of platform variability — for example unmeasured slack, inconsistent pretension, changing tendon seating, unvalidated tracking, or undocumented calibration state — then model error reflects the experimental platform as much as the underlying robot~\cite{rao_how_2021,camarillo_mechanics_2008}.

**Rationale**: "For example" frames the list as illustrative rather than claiming sources enumerate it. Drops Shihora and Raimondi from this citation (they don't address training data); Shihora belongs with S19/S24 instead. Keeps Rao (general platform-vs-model principle) and Camarillo (trial-to-trial variability).

---

## 7. S26 — Cosserat / FE cost framing

**Original**
> Cosserat rod and finite element models capture distributed mechanics more directly than PCC but at greater computational and parameter-identification cost~\cite{rao_how_2021,russo_continuum_2023,oliver-butler_continuum_2019}.

**Issue**: Rao addresses Cosserat (variable-curvature) vs PCC, not FE. Oliver-Butler reports their Cosserat ran ~1 s on unoptimized MATLAB and notes "optimized implementations… can run in real time" (L1257–1260), which complicates "greater computational cost."

**Suggested revision**
> Cosserat rod and finite element models capture distributed mechanics more directly than PCC, at the cost of greater parameter identification and, in many implementations, longer computation~\cite{rao_how_2021,russo_continuum_2023,oliver-butler_continuum_2019,raimondi_understanding_2024}.

**Rationale**: Softens "greater computational cost" to acknowledge real-time Cosserat exists; keeps parameter-identification claim (well supported); adds Raimondi as the in-folder example of FE-with-friction specifically.

---

## 8. S34 — Four-modality sensing enumeration

**Original**
> Shape sensing methods reported in recent surveys include electromagnetic tracking, fiber-optic shape sensing, intraoperative imaging-based reconstruction, and flexible electric or magnetic sensors embedded in the robot body~\cite{shi_shape_2017,sincak_sensing_2024}.

**Issue**: No single cited survey enumerates all four. Shi (abstract L23): FBG / EM / imaging. Sincak (abstract L31–37): electricity / magnetism / optics. Sincak does not cover imaging; Shi does not categorize flexible electric/magnetic sensors as a class.

**Suggested revision**
> Recent surveys report several shape-sensing modalities, including fiber-optic (FBG) shape sensing, electromagnetic tracking, and intraoperative imaging-based reconstruction~\cite{shi_shape_2017}, as well as flexible electric and magnetic sensors embedded in the robot body~\cite{sincak_sensing_2024}.

**Rationale**: Maps each modality to the survey that actually classifies it.

---

## 9. S39 — Grassmann citation scope

**Original**
> Experimental behavior is strongly shaped by fabrication tolerances, actuation hardware, routing details, friction, material response, and sensing integration~\cite{grassmann_open_2024,rao_how_2021,raimondi_understanding_2024}.

**Issue**: Grassmann covers fabrication tolerances and actuation/friction but does not enumerate routing, material response, or sensing integration as drivers.

**Suggested revision**
> Experimental behavior is strongly shaped by fabrication tolerances and actuation hardware~\cite{grassmann_open_2024}, by routing details, friction, and material response~\cite{rao_how_2021,raimondi_understanding_2024}, and by sensing integration~\cite{shi_shape_2017,sincak_sensing_2024}.

**Rationale**: Each citation now backs only the factors it actually covers.

---

## 10. S42 — Seven-element protocol attribution

**Original**
> Openness alone, however, does not guarantee repeatability: a useful experimental platform must also define how the robot is assembled, initialized, pretensioned, tracked, registered, commanded, and logged so that trials can be compared across time and across system configurations~\cite{grassmann_open_2024,rao_how_2021,russo_continuum_2023}.

**Issue**: Grassmann supports the general claim that openness alone is not enough (L76, L200–202, L820–824, L287–290) but does not itemize the seven protocol elements. That list is the thesis author's contribution.

**Suggested revision**
> Openness alone, however, does not guarantee repeatability~\cite{grassmann_open_2024}: a useful experimental platform must also define how the robot is assembled, initialized, pretensioned, tracked, registered, commanded, and logged so that trials can be compared across time and across system configurations.

**Rationale**: Attributes only the openness-insufficiency claim to Grassmann; treats the protocol-list as the thesis's own framing without an attached citation that would over-claim.

---

## 11. S15 (C3) — Camarillo extrinsic-actuation framing

**Original**
> They are attractive for benchtop research and surgical-scale prototypes because tendon actuation is extrinsic --- the actuators can be grounded off the robot so the distal manipulator remains compact and mechanically simple~\cite{rao_how_2021,camarillo_mechanics_2008,clark_continuum_2021}.

**Issue**: Rao L68–69 uses "extrinsic actuation" verbatim. Camarillo describes proximal standoffs but does not use the word "extrinsic" and does not state the design principle.

**Suggested revision**
> They are attractive for benchtop research and surgical-scale prototypes because tendon actuation is extrinsic --- the actuators can be grounded off the robot so the distal manipulator remains compact and mechanically simple~\cite{rao_how_2021,clark_continuum_2021}.

**Rationale**: Drops Camarillo from the extrinsic-framing citation. Camarillo is still cited heavily in adjacent sentences (S14, S16, S18, S19) where it does the work directly.

---

## 12. S3 — "motion scaling" (minor)

**Original**
> Robot-assisted platforms have addressed some of these limitations through improved visualization, motion scaling, and distal dexterity, but most established surgical robots remain constrained by the geometry and finite-joint structure of their instruments~\cite{dupont_continuum_2022,da_veiga_challenges_2020}.

**Issue**: "Motion scaling" is not in those exact words in Dupont (L194–197 lists "intuitive/autonomous control, ergonomics, hysteresis/friction compensation, image-based planning"). Concept is covered.

**Suggested revision (optional, conservative)**
> Robot-assisted platforms have addressed some of these limitations through improved visualization and ergonomic distal-dexterity control, but most established surgical robots remain constrained by the geometry and finite-joint structure of their instruments~\cite{dupont_continuum_2022,da_veiga_challenges_2020}.

**Rationale**: Removes the un-cited specific phrase "motion scaling" if you want strict source-fidelity. Alternatively leave the original — it's not wrong, just slightly looser than the cited words.

---

## 13. Add Wolfe to citations for hysteresis evidence (cross-section)

For sentences that discuss hysteresis (S24, S30, S33 if cited), consider adding Wolfe 2024 since Wolfe **empirically demonstrates** hysteresis on this very platform (1.181 mm cluster RMS on cable-axis points, ANOVA p ≈ 5×10⁻⁷). Wolfe is in the folder and is closer to the thesis topic than any of the general-mechanics sources.
