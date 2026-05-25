# AI/Agent Residue Scrub Report

Generated: 2026-05-24. Branch: `main` (HEAD `958e429`). Audit-only — no
files were modified.

Scope: full repo, excluding `.claude/`, `.git/`, `node_modules/`,
`__pycache__/`, `.venv/`, `.pytest_cache/`, `continuum_robot.egg-info/`,
`bin/`, `data/`, and read-only `references/` (per `AGENTS.md`).
`tools/`, `tools_local/`, and `legacy/` were spot-checked but contain
no AI residue.

Headline numbers:
- **133 commits** out of 332 total carry a `Co-Authored-By: Claude
  Opus 4.7 (1M context) <noreply@anthropic.com>` trailer (~40 %).
- **11 source files** contain at least one direct AI/Claude/Anthropic/
  ChatGPT/Codex reference.
- **5 scripts/workflows** are agent infrastructure rather than
  application code.
- **3 docs** are pure Claude/agent meta-content.
- **0** stylistically AI-ish comments, hedged docstrings, "I'll/we
  can/Note that" boilerplate, chatty `logger.info` strings,
  `TODO`/`FIXME`/`XXX`/`HACK` markers, "this code was generated"
  disclaimers, `example.com`/`your_name_here` scaffolding, or wrong-
  behavior docstrings were found.

The codebase reads as human-written project code throughout. The only
residue is structural: a handful of agent-infrastructure scripts/docs
and the commit-trailer attributions.

---

## 1. Direct references found

Each row: `file:line` → context (truncated) → suggested action.

### 1a. Agent infrastructure (intentional, not residue)

These files exist *to run Claude Code* on this repo. They are
operational scripts and CI for the author's workflow, not application
code. Going public means deciding whether the author wants the
workflow visible.

| File | Lines | What it is | Suggested action |
|---|---|---|---|
| `scripts/claude-bg.sh` | 1–84 (entire file) | Headless background runner for Claude Code | **Delete** for a clean public repo; **keep** if author wants the workflow public. |
| `scripts/claude-bg-stop.sh` | entire file | Companion to `claude-bg.sh` | Same as above. |
| `scripts/claude-parallel.sh` | entire file | Multi-worktree parallel Claude dispatch | Same as above. |
| `.devcontainer/devcontainer.json` | 1–40 (entire file) | Devcontainer for sandboxed Claude Code; installs `@anthropic-ai/claude-code` and `anthropic.claude-code` VS Code extension | **Delete** — devcontainer's only documented purpose is sandboxed AI work. |
| `.github/workflows/claude.yml` | 1–57 (entire file) | `@claude`-mention GitHub Actions integration using `anthropics/claude-code-action@v1` and `ANTHROPIC_API_KEY` secret | **Delete** — fully exposes the author's AI workflow + secret name to advisors. |
| `docs/HANDS_OFF_CLAUDE.md` | 1–87 (entire file) | Operator manual for running Claude Code 4 different ways | **Delete** — this is purely AI workflow documentation. |

`.github/workflows/tests.yml` is the legitimate CI workflow with no AI
references and should stay.

### 1b. Operator-facing prose that mentions ChatGPT (likely deliberate)

The author mentions "ChatGPT review" / "Mac or ChatGPT handoff" as a
real operator workflow — they intentionally hand exported bundles to
ChatGPT for review. This isn't AI-generation residue, it's a documented
sharing mode.

| File | Line | Context | Suggested action |
|---|---|---|---|
| `README.md` | 267 | "To package one run for a Mac, ChatGPT review, or advisor handoff…" | **Rephrase** to "advisor or external review" if going public; **keep** if the workflow stays. |
| `plans.md` | 169 | "Export bundles and the thesis evidence index exist for advisor/Mac/ChatGPT handoff and review." | Same as above. |
| `docs/architecture.md` | 187 | "build thesis evidence index for advisor/Mac/ChatGPT handoff" | Same as above. |

### 1c. Reference docs that name Claude/Codex

| File | Line(s) | Context | Suggested action |
|---|---|---|---|
| `docs/current_system_audit_claude.md` | 1, 3, 194 | Title "Skeptical System Audit (Claude)"; mentions "rushed-codex artifact"; references worktree `agitated-vaughan-8e590d` | **Rename to `docs/current_system_audit.md`** (drop "claude"), drop the worktree reference on line 3, change "rushed-codex artifact" → "rushed artifact" or "duplicate-write artifact". The audit content itself is human-readable analysis and worth keeping. |
| `docs/recent_data_quality_audit.md` | 3 | "Date: 2026-05-14. Worktree: `agitated-vaughan-8e590d`. Read-only audit." | **Edit line 3** — drop the worktree reference. |
| `docs/archive/2026-codex_seed_prompt.md` | 1, body | Archived original Codex bootstrap prompt; the first line declares it historical | **Keep but mark clearly** in `docs/archive/README.md`; or **delete** if you don't want the AI bootstrap history public. The doc itself is historical context, not generated code. |
| `docs/archive/README.md` | 13 | "AGENTS.md (repo root) — repo guidance for AI agents and contributors" | **Rephrase** to "AGENTS.md (repo root) — repo guidance / contributor protocol" to soften the AI focus. |
| `AGENTS.md` | 152 | "When the user reports a repeated misconception or recurring bug source, update `AGENTS.md` so the correction persists in future sessions." | The whole file is a "Repo Instructions" doc that doubles as agent guidance. The content is legitimate human-written repo policy (read-only paths, canonical runtime layout, working rules). **Rename to `CONTRIBUTING.md`** and rephrase line 152 ("…persists across contributor sessions" or "…stays canonical"). Most of the doc is genuinely useful contributor guidance regardless of who's reading. |

### 1d. Identifier residue inside application code

| File | Lines | What it is | Suggested action |
|---|---|---|---|
| `continuum_robot/gui/tabs/data_management_tab.py` | 288–292, 299–300, 366, 470–471, 903–907 | GUI buttons + handlers named `export_ai_selected_button`, `export_ai_latest_button`; profile string `"ai_debug"` | **Borderline**. The `ai_debug` export profile is a *documented operator feature* (a richer bundle for AI-assisted debugging) — see `continuum_robot/data/export_run_bundle.py:21,184,312,434,472` and `tests/test_export_run_bundle.py:139–152`. Renaming requires touching 5 files + button labels. Suggested: **keep** unless the author wants to disavow the AI-assisted debug-handoff workflow entirely. The naming honestly describes what the bundle is for. |

### 1e. Single mention in `.gitignore` adjacency

`.gitignore` does **not** include `.claude/`. The directory is currently
present in the working tree as untracked / via git submodules and is
not committed. Worth adding `.claude/` to `.gitignore` before going
public so it can never accidentally land. (Action: 1-line gitignore
add — not in scope for "remove residue" but flagged for completeness.)

---

## 2. Suspect docstrings / comments

**None found that match the "stylistically AI-written" or "describes
wrong behavior" criteria.**

Grep sweeps for:
- `Note that`, `I'll`, `I'd`, `we can simply`, `let me know`,
  `don't hesitate`, `hope this helps`, `Happy to`
- `It should be noted`, `It is worth noting`, `as we can see`,
  `It is important to note`
- `# I will`, `# We can`, `# We will`, `# Let us`, `# Note: this`

…returned **no matches** in application code. The few hits that did
turn up are domain-legitimate:
- `tests/test_mike_cc_math_invariants.py:182` — "We can verify by
  predicting…" — natural test prose.
- `continuum_robot/experiments/builtins.py:9010` — `# Note:
  renamed from "post_motion_telemetry_resync_success"…` — actual
  rename rationale.
- `continuum_robot/modeling/model_comparison.py:876` — `# Note:
  warnings are intentionally NOT drawn…` — design rationale.
- `continuum_robot/modeling/ann_training.py:3559` — `f"Note: {note}"` —
  literal output string.

The docstring style throughout the repo is dense, technical, and
project-specific. Examples sampled in `continuum_robot/experiments/
pair_axis_convention.py`, `continuum_robot/experiments/dataset_io.py`,
and `continuum_robot/servos/safety_guard.py` all read as authored
prose — no excessive hedging, no AI tics.

No `TODO`, `FIXME`, `XXX`, or `HACK` markers anywhere in the active
codebase (excluding `references/` and `legacy/`).

---

## 3. Files that should probably be deleted entirely

Sorted from "almost certainly delete" to "judgment call":

1. **`docs/HANDS_OFF_CLAUDE.md`** — pure Claude Code operating manual.
   Has no value to advisor / thesis reviewer / public audience.
2. **`.github/workflows/claude.yml`** — `@claude` GitHub App
   integration. Names `ANTHROPIC_API_KEY` and exposes the AI workflow
   in CI. Public visibility is a separate question from "want this
   running on PRs".
3. **`.devcontainer/devcontainer.json`** — devcontainer's documented
   purpose is sandboxed Claude Code; installs the `anthropic.claude-
   code` VS Code extension and `@anthropic-ai/claude-code` npm
   package. Without the AI workflow it has no other use.
4. **`scripts/claude-bg.sh`**, **`scripts/claude-bg-stop.sh`**,
   **`scripts/claude-parallel.sh`** — operational scripts for the
   author's headless Claude workflow. Not load-bearing for the robot.
5. **`continuum_robot/utils/math_utils.py`** — one-line file:
   `"""Math helper module placeholder."""`. No imports anywhere in
   the repo. Looks like an agent stub that was never filled in.
   **Delete** unconditionally.
6. **`docs/archive/2026-codex_seed_prompt.md`** (496 lines) —
   intentionally archived bootstrap prompt. The doc itself flags it
   as historical. **Keep** if you want the project's bootstrap
   history visible; **delete** if you'd rather the public not see how
   the codebase was originally LLM-bootstrapped.

Not flagged for deletion:
- `AGENTS.md` — rename + light edit (see 1c above); content is real
  contributor guidance.
- `docs/current_system_audit_claude.md` — rename + drop worktree
  reference (see 1c); analysis is human-readable and useful.
- `docs/recent_data_quality_audit.md` — drop the worktree reference
  on line 3.

---

## 4. Commit message audit

**Counts (across all branches):**
- Total commits: **332**
- Commits with `Co-Authored-By: Claude Opus 4.7 (1M context)
  <noreply@anthropic.com>` trailer: **133** (~40 %)
- Commits with `Claude` anywhere in the subject line: **9**

**Breakdown of Claude-attributed commits by conventional prefix:**

```
68  feat
32  fix
 7  refactor
 4  docs
 4  chore
 3  perf
 3  config
 2  two-segment
 2  test runner
 2  test
```

**Commits with `Claude` in the *subject* (9 total):**

```
9efdf0a On claude/blissful-sutherland-5c66b0: abandoning blissful-sutherland-5c66b0
1395e9f index on claude/blissful-sutherland-5c66b0: cf67d79 fix(tracker_timing): ...
3deac89 untracked files on claude/blissful-sutherland-5c66b0: cf67d79 fix(tracker_timing): ...
9266265 feat(devcontainer): add devcontainer configuration for sandboxed Claude Code work
        feat(github-actions): implement Claude integration workflow ...
        docs: create HANDS_OFF_CLAUDE.md for hands-off usage instructions of Claude Code
        feat(scripts): add background and parallel execution scripts for Claude Code tasks
024afb2 Merge remote-tracking branch 'origin/main' into claude/mystifying-kalam-341aa5
dfcc907 Merge remote-tracking branch 'origin/main' into claude/mystifying-kalam-341aa5
0fcb31e Merge remote-tracking branch 'origin/main' into claude/mystifying-kalam-341aa5
27926ce Merge claude/amazing-lovelace-0da0e6: two-segment foundation (40 cycles)
1e5822b Merge remote-tracking branch 'origin/main' into claude/amazing-lovelace-0da0e6
```

The first three are `git stash` entries from a worktree
(`9efdf0a`, `1395e9f`, `3deac89`) — these can be cleaned up with
`git stash drop` per branch but are not on `main`.

The five `Merge … claude/<adjective>-<noun>-<hex>` subjects are
genuine merge commits from per-worktree branches into main; they
expose the worktree naming scheme.

`9266265` is the commit that introduced the Claude-infra files
(devcontainer, workflow, scripts, docs/HANDS_OFF_CLAUDE.md).

**Sample of standard Claude-attributed commits** (first 5 most
recent on `main`):

```
454c3b6 two-segment: top-tendon routing compensation + 2 s settle default
1f945a8 config: two-segment bench-up defaults (slower servos, B=bottom, single coil)
e447a93 feat(experiments): thesis-eligibility verdict stamping
cf67d79 fix(tracker_timing): simplify figure set per operator request
709bdf2 fix(experiments-tab): surface neutral_setpoint_drift_validation in the GUI dropdown
```

These subject lines themselves are clean engineering messages — the
only attribution is the trailer in the commit body.

**No rewrite is recommended in this pass.** The user explicitly said
"note these but do not rewrite history yet — that's a separate
decision." If a rewrite is later approved, the cleanest path is:

```
git filter-repo --message-callback '
  return re.sub(rb"\n+Co-Authored-By: Claude.*?\n", b"\n", message)
'
```

…run from a fresh clone, then force-pushed. Be aware:
- All 133 commits change SHAs.
- Any outstanding PRs, advisor links, or `.dat` files referencing
  commit hashes get invalidated.
- Worktrees and stashes on `.claude/worktrees/*` will need re-syncing.

A lighter-weight alternative: **leave history alone** and only scrub
forward (i.e., stop adding the trailer in new commits). Advisors
generally don't audit git history; the public-facing surface is
mostly file content + tip-of-main, not trailers.

---

## 5. Items I'm unsure about

These are genuine judgment calls — flagging for review rather than
recommending an action.

1. **`AGENTS.md` (repo root)**. The file's content is real, valuable
   contributor guidance (read-only path policy, canonical runtime
   layout, working rules, validation expectations). The framing is
   "agent + contributor" — see line 13 of `docs/archive/README.md`.
   Options:
   - **Rename to `CONTRIBUTING.md`** + soften one line (152) and keep
     everything else.
   - **Keep as `AGENTS.md`** — it's a standardized convention some
     repos use, and the content stands on its own.
   - **Split** into `CONTRIBUTING.md` (working rules, paths) +
     `AGENTS.md` (the bits that are explicitly about LLM agents).
   No wrong answer; I'd lean "rename to `CONTRIBUTING.md`" for a
   public thesis repo.

2. **`continuum_robot/data/export_run_bundle.py` profile name
   `ai_debug`** (line 21, 184, 312, 434, 472) and its GUI surface
   (`continuum_robot/gui/tabs/data_management_tab.py`). Renaming is
   load-bearing — it touches the profile string in the manifest, the
   CLI flag, the GUI button labels, and the test assertions. Decision
   hinges on whether the author wants to keep "AI-assisted debug
   handoff" as a first-class documented profile or rename it to
   something neutral like `full_debug`. Functionally identical either
   way.

3. **`docs/archive/2026-codex_seed_prompt.md`**. The file is honest
   about being a historical Codex prompt. Public visibility is fine
   if the thesis story is "this was bootstrapped from an LLM seed
   prompt and then iterated"; less fine if the thesis story is "this
   was authored from scratch". Author's call.

4. **`docs/current_system_audit_claude.md`**. The filename says
   "Claude" but the content is a normal self-audit ("Skeptical System
   Audit") that any senior engineer could have written. Renaming to
   `docs/current_system_audit.md` and dropping the worktree reference
   on line 3 makes it disappear into the doc set; deleting it loses
   real findings (F-1 through F-12 are concrete, useful bug-list).
   I'd rename rather than delete.

5. **`continuum_robot/utils/math_utils.py`** — flagged for deletion
   in §3. The only reason to keep it would be if there's an
   intentional future home for math helpers. Currently zero callers.

6. **The `.claude/` directory not being in `.gitignore`**. Out of
   scope for "remove residue" but easy belt-and-suspenders before
   going public. Adding `.claude/` to `.gitignore` prevents any
   future accidental commit of session state, settings, or worktree
   metadata. Strictly safer than the status quo, no downside.

---

## Decision matrix (for review)

| # | Item | If you want **maximally clean public repo** | If you want **transparent about LLM use** |
|---|---|---|---|
| 1 | `docs/HANDS_OFF_CLAUDE.md` | delete | keep |
| 2 | `.github/workflows/claude.yml` | delete | keep |
| 3 | `.devcontainer/devcontainer.json` | delete | keep |
| 4 | `scripts/claude-bg*.sh`, `claude-parallel.sh` | delete | keep |
| 5 | `continuum_robot/utils/math_utils.py` | delete | delete |
| 6 | `docs/archive/2026-codex_seed_prompt.md` | delete | keep |
| 7 | `docs/current_system_audit_claude.md` | rename | rename |
| 8 | `docs/recent_data_quality_audit.md` (line 3) | edit | edit |
| 9 | `AGENTS.md` | rename to `CONTRIBUTING.md`, edit l.152 | keep, edit l.152 |
| 10 | `README.md` line 267, `plans.md` line 169, `docs/architecture.md` line 187 ("ChatGPT review") | rephrase to "external review" | keep |
| 11 | `docs/archive/README.md` line 13 ("AI agents and contributors") | rephrase | keep |
| 12 | `data_management_tab.py` `ai_debug` profile + buttons | rename to `full_debug` | keep |
| 13 | `Co-Authored-By: Claude` trailers in 133 commits | rewrite history (cost: SHAs change) | leave |
| 14 | Add `.claude/` to `.gitignore` | yes | yes |

Standing by for a list of items to act on.
