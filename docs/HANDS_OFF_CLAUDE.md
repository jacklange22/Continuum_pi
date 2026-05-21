# Hands-off Claude on `pi_code`

Four ways to run Claude Code on this repo without babysitting it, ranked roughly from "right now, this laptop" to "fully remote, no machine of mine involved."

## 1. Local background runs — `scripts/claude-bg.sh`

Best for: small-to-medium tasks you want to fire and forget while you do something else on the same laptop.

```bash
scripts/claude-bg.sh "fix the failing test in tests/registration/test_persistence.py"

# come back later:
scripts/claude-bg-stop.sh             # list active runs + their log paths
tail -f .claude/bg-logs/bg-*.log      # watch one in real time
scripts/claude-bg-stop.sh <tag>       # kill one
scripts/claude-bg-stop.sh --all       # kill them all
```

Under the hood: `claude -p --permission-mode bypassPermissions --output-format stream-json`, detached via `nohup` so closing the terminal doesn't kill it. Logs land in `.claude/bg-logs/<tag>.log`.

## 2. Parallel attempts across worktrees — `scripts/claude-parallel.sh`

Best for: hard problems where you want 3 different attempts and to pick the best diff. You already use `.claude/worktrees/`, this just fans the dispatch out.

```bash
# Three fresh worktrees, same prompt, three independent attempts.
scripts/claude-parallel.sh -n 3 "refactor continuum_robot/tracking to remove the legacy NDI shim"

# Run different prompts in different existing worktrees.
scripts/claude-parallel.sh \
  -w funny-tharp-199d99 "add type hints to continuum_robot/servo/" \
  -w great-ride-96aa7c  "write tests for continuum_robot/registration/exporters.py"

# Diff the winner back to main:
git -C .claude/worktrees/par-7f3a91 diff main
```

## 3. Sandboxed YOLO mode — devcontainer

Best for: tasks where you want `--dangerously-skip-permissions` (Claude never asks) but don't want a stray `rm -rf` to touch your Mac.

```bash
# One time:
brew install --cask docker
npm install -g @devcontainers/cli

# Build & enter the sandbox:
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . claude --dangerously-skip-permissions \
  "implement the missing exporters and run the test suite until it passes"
```

The container at `.devcontainer/devcontainer.json` mounts only this repo. Filesystem writes outside `/workspaces/pi_code` are invisible to Claude. Combine with `scripts/claude-bg.sh -p bypassPermissions` inside the container for true overnight runs.

## 4. Fully remote — GitHub Actions `@claude`

Best for: tasks you want to kick off from your phone, or while your laptop is asleep.

One-time setup, from inside `claude` on this repo:

```text
/install-github-app
```

That installs the Anthropic GitHub App and sets the `ANTHROPIC_API_KEY` secret. After that, just write `@claude please add a regression test for the tracking timestamp bug` in any issue or PR comment — the workflow in `.github/workflows/claude.yml` runs Claude on a GitHub runner and opens/updates a PR.

Triggers wired up:
- new issue body or title containing `@claude`
- any issue / PR / review comment containing `@claude`
- issue assigned (extend the workflow if you want this to auto-trigger)

## Safety nets in this repo

`AGENTS.md` already marks `tools/` and `references/` as read-only. Belt-and-suspenders:

- `.claude/settings.json` — your existing allow list and `bypassPermissions` default.
- `.claude/settings.local.json` (new, gitignored by default in most setups — check yours) — explicit `deny` rules for `tools/`, `references/`, `rm -rf`, `sudo`, `git push --force`, `git reset --hard`, `curl | sh`, etc. These deny rules override the bypassPermissions default, so even YOLO mode can't trip them.

## Which one should I actually use?

| Situation | Use |
|---|---|
| "Run this while I take a meeting." | `claude-bg.sh` |
| "Three approaches, pick the best diff." | `claude-parallel.sh -n 3` |
| "Run overnight, don't ask me anything." | devcontainer + `claude-bg.sh` inside |
| "Do this from my phone / no laptop." | `@claude` in a GitHub issue |

