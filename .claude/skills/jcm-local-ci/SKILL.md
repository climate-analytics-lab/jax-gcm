---
name: jcm-local-ci
description: Run the jax-gcm CI gates locally on Derecho when GitHub Actions minutes are exhausted or a pre-push check is wanted — lint, fast tests (90% coverage), slow tests (80% PR coverage) and a local Claude code review. Use before pushing or merging any jcm branch without Actions.
---

# Local CI for jax-gcm

Reproduces the three CI gates from `.github/workflows/run_test.yaml` +
`run_linter.yaml`, plus the Claude review, without GitHub Actions.

## One command

```bash
scripts/local_ci.sh /path/to/worktree     # lint + fast gate locally, slow gate via qsub
```

It runs lint and the fast gate on the current node (~3–5 min with
`-n 12`) and submits the slow gate as a `develop`-queue PBS job
(~30–45 min). Watch the job log for `SLOW_EXIT=0`.

## The gates, individually

```bash
# 1. Lint (run_linter.yaml)
ruff check .

# 2. Push gate — fast tests, 90% coverage
JAX_PLATFORMS=cpu pytest -n 12 -m "not slow" --cov=jcm --cov-fail-under=90

# 3. PR gate — slow tests only, 80% coverage vs .coveragerc-pr
JAX_PLATFORMS=cpu pytest -n 4 -m "slow" --cov=jcm \
    --cov-config=.coveragerc-pr --cov-fail-under=80
```

Gate 3 is real compute — run it on a `develop`-queue node
(8 cpus / 220 GB, ~13 min with `-n 4`), not a login node. The `-n 4` is
**mandatory on Derecho**, not an optimization: a single serial process
accumulates thousands of mmap'd XLA JIT code sections and dies mid-suite
with `LLVM ERROR: Unable to allocate section memory!` (SIGSEGV/SIGABRT —
three attempts at 120–220 GB all failed identically; RAM is not the
issue, per-process map count is). Splitting across workers resets the
budget. GitHub's runners tolerate the serial run; Derecho's do not.

## Local Claude review

In a Claude Code session on the branch: `/code-review high` reviews the
diff vs upstream with multi-agent finders + verification — no Actions
minutes, billed to the Claude session. For the deep cloud variant use
`/code-review ultra` (user-triggered, separately billed).

## Codex review comments: always reply inline

Every Codex (or other bot) inline comment on a PR gets an **explicit
threaded reply** stating the resolution — "Confirmed and fixed in
<commit>: <what changed>" or "Refuted: <the evidence>" — via

```bash
gh api -X POST repos/<owner>/<repo>/pulls/<PR>/comments/<comment_id>/replies \
    -f body="..."
```

(comment ids from `gh api repos/<owner>/<repo>/pulls/<PR>/comments`).
Fixing the code silently is not enough: the reviewer tracks resolution
through the comment threads, and an unanswered thread reads as an
unaddressed finding. Verify a claim against the actual code/data before
replying — Codex has been right (forcing unit conventions) and wrong
(FZJ ozone units) on the same PR.

## Facts that bite

- **Login-node load produces phantom failures.** A `-n 12` fast-suite
  run on a busy login node has produced 50+ failures across unrelated
  subsystems that all pass serially (twice now: 55 in July, 53 in
  August). Before believing a red local run, rerun a sample of the
  failures serially; better, don't share the node with heavy I/O jobs
  during the run.
- **Capture pytest's exit via `PIPESTATUS[0]`**, never `$?` after a
  `| tail` — and never pipe the slow suite through `tail` at all (a
  segfault's context ends up truncated).
- **Lint the whole repo (`ruff check .`), never a subdirectory** — CI
  lints everything, including tools/ and stray root files. And never
  commit with `git add -A`: it sweeps untracked scratch files into the
  commit, which is exactly how a lint-clean `jcm/` still turned CI red
  once. Stage files by name.

- **Coverage differs by suite.** The fast gate uses `.coveragerc`
  (builders omitted); the slow gate uses `.coveragerc-pr` (also omits
  fast-tested utility modules). New fast-tested modules that slow tests
  never touch belong in `.coveragerc-pr`'s omit list — that is the
  repo's documented mechanism, see the header comment there.
- `JAX_PLATFORMS=cpu` is mandatory on GPU nodes (xdist workers
  otherwise fight over the GPU).
- The repo pins `ruff` in CI; run the pinned version (`pip show ruff`
  vs `.github/workflows/run_linter.yaml`) before trusting a clean pass.
- GPU-gated slow tests skip on CPU exactly as they do in CI — a local
  CPU pass is equivalent evidence.
