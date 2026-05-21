# Audit Log

## What was audited

The Background section provided inline, comprising ~44 sentences across 9 paragraphs plus one comparison table with 9 data rows. Total citation instances analysed: ~80.

## What was read

All 20 .txt files in `/Users/jacklange/Downloads/txtResearch/` were accessible and readable. None failed to parse. File-level verification was performed via grep + targeted Read on each cited source.

| Source | Read mode | Coverage |
|---|---|---|
| Raimondi 2024 | Deep (numerical claims, qualitative claims) | Abstract + Methods + Results + Conclusion |
| Wolfe MS Thesis 2024 | Deep (numerical claims, hysteresis discussion) | Abstract + Methods + Results + Conclusion |
| Camarillo 2008 | Deep (mechanics, extrinsic framing, hysteresis) | §I–§V |
| Webster & Jones 2010 | Deep (PCC, decomposition, tendon guides) | Abstract + §3 + §6 |
| Rao 2021 | Deep (tendon-driven, Cosserat, learning, calibration) | §1 + §5 + §9 |
| Shihora & Simaan 2024 | Deep (FLER framework, friction, design-stage framing) | Abstract + §1 + §4 + §6 |
| Li 2024 | Deep (7.89 mm number — verified as simulation, not prototype) | Abstract + §3.2 + §4.2 + Fig. 8 caption |
| Oliver-Butler 2019 | Targeted (Cosserat, tendon stretch, loaded shape, real-time feasibility) | Abstract + §IV + §V |
| Burgner-Kahrs 2015 | Targeted (MIS motivation, definition, compliance) | §1, §2, §6 |
| Dupont 2022 | Targeted (MIS, taxonomy, Ion/Monarch, robotics benefits) | §1, §2, §3 |
| Da Veiga 2020 | Targeted (MIS, rigid limits, taxonomy, fluidic/magnetic) | §1, §3, §5 |
| Zhong 2020 | Targeted (architecture taxonomy, each family) | §2, §3 |
| Russo 2023 | Targeted (definition, taxonomy) | §1, §2 |
| Shi 2017 | Targeted (three-category enumeration, EM registration, trade-offs) | Abstract + §I + §V |
| Sincak 2024 | Targeted (three-category enumeration, challenges) | Abstract + §3 + §6 |
| He 2025 | Targeted (EM transform, calibration, comparative trade-offs) | §3, Table 1, §6 |
| Grassmann 2024 | Targeted (custom-prototype statistics, OCRP name, reproducibility) | §1, §2, §6 |
| Clark 2021 | Targeted (ENDO name, open-source motivation) | Abstract + §1 |
| Sozer 2023 | Not read (not cited in Background) | — |
| Continuum_Robot_Stiffness… | Not read (duplicate of Oliver-Butler 2019) | — |

## How the audit was performed

1. **Folder inspection and file inventory** — confirmed all 20 .txt files exist and are non-empty; sizes 21 KB to 209 KB; line counts 430 to 5,126.
2. **Section decomposition** — Background section parsed into ~44 sentence-level claims plus 9 table rows. Each claim labelled with its citation set.
3. **Cluster assignment** — claims grouped into 5 verification clusters by primary cited source, then dispatched to parallel sub-agents:
   - Cluster R/W: Raimondi-2024 numerical claims + Wolfe-2024 numerical claims
   - Cluster CWR: Camarillo 2008 + Webster 2010 + Rao 2021 modeling-foundation
   - Cluster CL: Clinical/MIS — Burgner-Kahrs + Dupont + Da Veiga + Zhong + Russo
   - Cluster SR: Sensing/Registration — Shi + Sincak + He + open-source platform (Grassmann + Clark)
   - Cluster SLO: Shihora + Li + Oliver-Butler
4. **Per-claim verification** — each sub-agent grep'd target terms in the cited source(s), Read the surrounding context, and rated support on the scale Strong / Partial / Weak / Indirect / Not Found / Contradicted. Sub-agents were instructed to be skeptical and quote source text verbatim with line-number context.
5. **Aggregation** — sub-agent reports synthesised into the 5 deliverable files.

## Assumptions and limitations

- **PDF-to-text artefacts**: All sources are .txt extracted from PDFs (size profile and line counts confirm OCR-style extraction). Line numbers are .txt-file line numbers and approximate; they should be treated as locators for the grep query, not authoritative page citations. Equations, figures, and tables in source PDFs were not directly inspected — only the text representation.
- **Citation key inference**: The thesis uses citation keys like `dupont_continuum_2022` and `burgner-kahrs_continuum_2015`. These were mapped to filenames by year, first author, and title slug. Mapping was unambiguous in every case.
- **No bibliography file inspected**: The audit verified citations against source content, not against any `.bib` file. If the underlying BibTeX entries have inconsistent author/year metadata, that's outside scope.
- **Sozer 2023 not audited**: present in folder but not cited in the Background section. If it is cited elsewhere in the thesis (e.g., §3.2 spine design or §3.3 base design), this audit does not cover that.
- **Duplicate file**: `Continuum_Robot_Stiffness_Under_External_Loads_and_Prescribed_Tendon_Displacements.txt` is byte-identical (77 821 bytes, 1 717 lines) to `Oliver-Butler et al. - 2019 - …`. Treated as one source.
- **Topic-sentence claims**: Several sentences (S5, S17, S20, S31, S33, S37, S38) are interpretive bridges or topic sentences without citations. They were noted in the "could-be-cited" list but not flagged as failures, since they are author-position statements.
- **Wolfe thesis self-citation**: The thesis-author is auditing citations to Wolfe 2024, which is a prior thesis from the same lab. The audit treated Wolfe's numerical claims as it would any external source — they checked out.
- **Sub-agent token limits**: Each sub-agent had a soft word ceiling of 1500 in its report. Where a cluster was very rich (Cluster CL: 13 claims × 5 sources), agents prioritised the highest-stakes claims (named-product fact-checks) and gave shorter coverage of fully-attested topical claims. Strong ratings in such cases reflect at least one verbatim quote from at least one cited source — not exhaustive coverage of every cited source.

## What was NOT verified (honest scope statement)

- **Bibliography metadata** (author lists, journal names, page numbers, DOIs) — the audit verified content, not citation-string formatting.
- **Cross-references inside the thesis** to figures, sections, or appendices — out of scope.
- **The thesis's table number "Table 2.1" / cross-references to other chapters** — out of scope.
- **Whether the cited sources are the best in the literature** — the audit verified that cited sources support attached claims; it did not search for stronger alternative sources. (Where a citation is clearly weak, the audit suggested either dropping it or adding a candidate replacement from the in-folder set.)
- **Numerical re-derivation** — Raimondi's % errors and Wolfe's mm errors were verified to appear in the source text but were not re-computed from raw experimental data.

## Final summary (printed)

- **Sentences audited**: ~44 prose sentences + 9 table rows = ~53 atomic claims
- **Citation instances checked**: ~80
- **Strongly supported**: ~33 claims (about 62%)
- **Partially or weakly supported**: ~14 claims (about 26%)
- **Overstated / mismatched**: 6 claims (S15 Camarillo for extrinsic, S19 pretension, S24 routing-losses + Rao on compression/hysteresis, S26 FE/cost framing, S28 Wolfe hysteresis-capture, S39 Grassmann enumeration, S42 Grassmann protocol checklist)
- **Contradicted**: 1 (TBL-Li attributes a simulation number to "prototype tests")
- **Citation needed (no current citation)**: 0 strict failures; 7 opportunities to add a citation on a topic-sentence or interpretive claim
- **Source unreadable**: 0
- **Sources unused**: 1 (Sozer 2023, present in folder but not cited in Background)

## Most important fixes before submission (priority-ordered)

1. **TBL-Li**: change "(prototype tests)" → "(simulation, avg across three paths)". The 7.89 mm is a simulation number, not a prototype number.
2. **S28**: soften "Learning-based models can capture… history-dependent behavior, including hysteresis" — Wolfe's ANN does NOT model hysteresis; Wolfe identifies it as a future-work direction.
3. **S27 / TBL-Raimondi-PCC**: change "4.10%" → "4.1%" (matches source precision).
4. **S19**: either drop "pretension" from the factor list or add Oliver-Butler/Wolfe (the only in-folder sources that touch tendon-tension state under loading).
5. **S24**: drop "routing losses" or replace with "tendon-path friction"; consider removing Rao from this citation as Rao does not cover backbone compression or material hysteresis.
6. **S15**: drop Camarillo from the "extrinsic actuation" citation — Rao is the right cite.
7. **S39**: restrict Grassmann citation to fabrication-tolerance + actuation portion; rely on Rao + Raimondi (+ optionally Shi/Sincak) for the remaining factors.
8. **S34**: split the four-modality citation so each modality is attached to a survey that actually enumerates it.
9. **S42**: attribute only the openness-insufficiency claim to Grassmann; treat the seven-element protocol checklist as the thesis's own synthesis.
10. **S30**: frame the platform-factor list as illustrative ("for example…") rather than implying source enumeration; consider dropping Shihora from this sentence (Shihora doesn't discuss learning/datasets).
