# Enforcing the living model description

`docs/source/science/` states every consequential scientific choice, its
provenance, and the gaps that remain. Prose rots silently: a renamed symbol
leaves a pointer that reads authoritative and points nowhere, and a cited
issue that has since closed turns a documented limitation into a claim the
reader believes was fixed. Two mechanisms hold the page honest, and they are
deliberately different in kind.

## Tree-deterministic checks block the pull request

`docs/science_pointers_test.py` resolves every ``file.py::symbol`` pointer
against the AST, every directory pointer against the tree, and every page
against the toctree. These are pure functions of the commit: the change that
breaks a pointer is the change that fails, and the person who caused it is the
person who sees it. They run in the ordinary suite and in the
`science-register-guard` job, which covers the doc-only pull requests the main
suite's `paths-ignore` skips.

## The issue-state check runs out of band

Whether a cited `#NNN` is still open is *not* a function of the tree, and that
difference matters more than it first appears. As a blocking test it made
closing an issue a breaking change for `dev`:

- a green merge went red with no commit touching the repository;
- pytest collects the guard whenever the suite runs, so the blast radius was
  every pull request touching `jcm/`, not just documentation ones;
- and the failure reached the next contributor, who has neither the context
  nor the standing to decide what the page should now say, while the person
  who closed the issue — who has both — saw nothing.

It happened twice (#825, #791) before being moved.

So `.github/workflows/science_register.yaml` enforces it on the two triggers
where it is actionable. On `issues: closed` it checks whether the register
cites that number and, if so, comments on the issue and files an assigned
follow-up — the signal reaches the closer while they still remember why they
closed it. A daily sweep catches what the hook cannot: issues transferred or
deleted, closures that happened while the workflow was failing, and citations
of numbers that never existed. Neither trigger can fail a pull request.

`docs/tracked_gaps.py` holds the parsing and the API lookup, so the guard, the
hook and the sweep cannot disagree about what counts as a citation. Running
the check locally is opt-in: `JCM_CHECK_TRACKED_GAPS=1`.

## Why not the obvious alternatives

**Let a page cite a closed issue** — a `#791 (closed)` marker, or a snooze
list of acknowledged-stale references. Both restore a green build by weakening
the invariant the guard exists to hold, and leave the page telling a reader
something untrue.

**Distinguish why the issue closed** — fixed, superseded, not planned. It does
not help: in every case the prose is now stale and wants an edit. #790 closed
because it was fixed and #791 because the plan changed, and both pages needed
rewriting.

**Check only the references a pull request adds.** This would keep a typo'd
number blocking at the point it is introduced, which is genuinely attractive.
It is not done because it needs diff parsing inside the guard and a base ref
that only exists in CI, and the daily sweep already reports such a typo within
a day. If the sweep proves too slow a signal in practice, this is the thing to
add.
