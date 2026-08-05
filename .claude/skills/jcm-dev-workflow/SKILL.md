---
name: jcm-dev-workflow
description: End-to-end development workflow for jcm — atomic commits, the local test/lint gate, opening a PR linked to its issue, monitoring CI *and* the automatic Codex review, addressing feedback, and handing back for human review only once everything is green. Use for any code change destined for a PR.
---

# Development workflow

The loop, in order. Each step exists because skipping it costs a CI cycle, a
review round, or a wrong result.

```
branch → atomic commits → test + lint locally → push → PR (linked to issue)
   → monitor CI *and* Codex → fix/respond → re-review if substantial → ping human
```

## 1. Work locally, committing atomically

Branch off `dev` (not `main` — clean releases are merged to `main` and
tagged). Never commit directly to `dev` or `main`.

One logical change per commit, each self-contained and passing tests on its
own. A commit message says **why**, not what the diff already shows: the
reference formulation being matched, the failure that motivated a guard, the
alternative rejected and the reason. When a change encodes a scientific
decision, that reasoning belongs in the commit *and* in a comment — see the
"Think Before Coding" and documentation rules in `CLAUDE.md`.

Stacking several subtasks on one PR is fine and often easier to review than a
chain of dependent PRs — say so in the PR body so the reviewer knows the
commits are separable.

## 2. The local gate — run this before every push

```bash
ruff check .                          # MUST be clean; it is the only linter
JAX_PLATFORMS=cpu pytest -n 12        # full suite, ~2 min
JAX_PLATFORMS=cpu pytest -n 12 -m "not slow"   # fast subset while iterating
```

**`JAX_PLATFORMS=cpu` is required** on this GPU host. Without it every xdist
worker grabs the same GPU and you get `CUDA_ERROR_OUT_OF_MEMORY` /
`dnn_support != nullptr` `RET_CHECK` failures from XLA. The unit tests are
small column-mode integrations that run faster on CPU than they would
round-tripping through the device.

- `-n 12` is the local default; `-n auto` picks from visible CPUs; `-n 0` (or
  omitting `-n`) forces one process when you need ordered output or are
  chasing a flake.
- Coverage: `JAX_PLATFORMS=cpu pytest -n 12 --cov=jcm --cov-fail-under=90`.
- Tests are `*_test.py`, co-located with the module, `unittest.TestCase` run
  under pytest. Root `conftest.py` clears `jcm` imports between tests to stop
  state leaking.
- Mark tests over ~1 min `@pytest.mark.slow`.

**CI thresholds**: push runs fast tests at **90%** coverage; a pull request
also runs the slow tests at **80%**. So a PR can surface slow-test failures a
push never did — run the full suite locally before opening one.

**Lint before every push, always.** `ruff check .` takes seconds; a lint
failure in CI burns a full cycle on something reported instantly locally.
Treat it as part of the definition of done.

Beyond the suite: a change to physics or numerics needs a **representative
run**, not just green tests. Verify conservation/physical correctness, and for
anything performance-related use `jcm-benchmark` — never quote the cumulative
`sim days/hr` line.

## 3. Push and open the PR

```bash
git push -u origin <branch>
gh pr create --base dev --title "..." --body "..."
```

**Link the issue** so it closes on merge: `Closes #NNN` (or `Refs #NNN` for
partial work) in the body. If the PR covers a subtask of a tracking issue, say
which.

Write the body for the reviewer: what changed, **why this approach over the
alternatives**, the decision anyone would second-guess, measured cost for
anything performance-affecting, and what you verified. Implementation-specific
detail and gotchas go here; general design goes in `docs/source/design/` per
`CLAUDE.md`.

## 4. Monitor CI *and* Codex — both, from one watcher

A Codex review arrives **automatically** on push; it is not triggered by you.
Watch for both, because green CI with unaddressed Codex findings is not done.

```bash
# CI checks
gh pr checks <PR> --watch

# Codex posts as a REVIEW with inline comments, not as an issue comment --
# `gh pr view --json comments` will NOT show it.
gh api repos/climate-analytics-lab/jax-gcm/pulls/<PR>/reviews \
  -q '.[] | "\(.user.login) | \(.state) | \(.submitted_at)"'
gh api repos/climate-analytics-lab/jax-gcm/pulls/<PR>/comments \
  -q '.[] | "### \(.path):\(.line)\n\(.body)\n"'
```

Arm a persistent `Monitor` that polls both and emits a line per new check
result or new review comment, exiting when checks are complete *and* a review
has landed. Make the filter cover **failure signatures too** — a watcher that
matches only success is silent through a crash, which looks exactly like
"still running".

## 5. Fix CI, respond to Codex

Fix every CI failure. Do not re-push hoping it was a flake; if you believe it
is one, say why.

For each Codex finding: **verify it against the code before acting.** They are
usually right and often catch real defects, but not always — confirm the
failure it describes is reachable. Verify against the *code*, not against
your notes or memory: a finding that contradicts something you believe is
the highest-value kind, because one of the two is stale and it may not be
the finding. Then either fix it, or reply explaining
concretely why it does not apply. Never silently ignore one.

Findings are labelled P1/P2/P3. P1s block. Reply on the PR summarising what
you changed for each, so the human reviewer can see the loop closed without
re-deriving it.

**Fix the class, not the instance.** Before calling a finding done, grep for
the same mistake elsewhere — a reviewer reports the instance they happened to
look at, not every occurrence. This is the single most common way a review
round gets repeated: in one session the *same* review loop re-reported a
NaN-check hole fixed in one file and left in another, and an
`sys.executable`-vs-run-interpreter assumption fixed in one script and left
in its sibling. Both were avoidable with one grep.

**Enumerate the category; do not grep for the forms you were shown.** This
is the part that is easy to get wrong, and getting it wrong looks exactly
like having done the sweep. Grepping for the specific strings in the review
comment finds only what you already knew about.

The same session ran three rounds of one defect — diagnostics reading
step-start state under operator splitting — because each sweep searched the
reported spellings (`state.tracers`, then `clouds.qc`) instead of asking
"which prognostic fields does any diagnostic read?". The third round finally
enumerated *every* `state.<field>` access in *every* diagnostics-category
term and produced a table with a verdict per site. That table is the
deliverable; a grep hit-count is not.

So: name the invariant ("a diagnostic must report the state as saved"), list
every place that invariant could be violated, and check each — including the
ones that turn out fine, because "already correct" is a result worth
recording. If your fix added a mechanism, check you consumed *all* of it: in
that session the new accumulator already carried the wind and humidity
tendencies that the next round's findings were about.

## 6. Re-request review when the response is substantial

If the fixes were more than trivial, ask for another pass with a comment
containing exactly:

```
@codex review
```

Then monitor again (step 4). Repeat until a review comes back with nothing
material.

## 7. Hand back to the human only when green

Ping the user for review once **all** of:

- CI checks pass
- Codex has no outstanding material findings
- `ruff check .` is clean and the full suite passes locally
- docs updated in the same PR if user-facing behaviour changed (`CLAUDE.md`:
  documentation lives with the change)

Then summarise: what landed, what was measured, what you decided and why, and
anything you are still unsure about. Flag open questions explicitly rather
than letting them pass as settled — an unflagged uncertainty is worse than a
known gap.

## Related skills

- `jcm-run` — driving the model (config groups, Hydra traps)
- `jcm-benchmark` — measuring throughput without fooling yourself
- `devbox-jcm-runs` / `derecho-jcm-runs` — machine-specific execution
