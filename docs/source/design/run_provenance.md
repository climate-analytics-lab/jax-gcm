# Run provenance: the record lifecycle and the best-effort contract

`jcm.provenance` records what produced an output file (code versions and
dirty state, precision and hardware, the boundary files opened, the parameter
values the run used, one run hash) as `jcm_prov_*` netCDF attributes and a
`<output>.provenance.json` sidecar. This page states the two contracts the
module keeps; the module docstring lists what is recorded.

## The record opens at the first thing recorded

`start_run(cfg)` is the only call that empties the input and fact registries.
Anything else that needs an open record (`record_input`, `record_fact`,
`collect`) opens one *without clearing*. The Hydra entry point calls
`start_run(cfg)` before it builds the model, but a model built from Python
never does, and its build already records inputs and facts (a forcing checked
against the run's dates records `dated_input_persistence`). A record that
opened by clearing would drop exactly those entries from the attributes, the
sidecar and `run_hash`, so two runs differing only in an early-recorded input
would share a hash. `created` is therefore the time of the first thing
recorded, and a second `start_run()` in the same process starts a clean
record.

## Probes are best-effort and cannot fail a run

Provenance is collected at output time, inside a run. A failure to *describe*
the run must never stop it. Every probe that touches the environment
(`git`, `importlib.metadata`, `getpass`, `socket`, device enumeration, input
files) therefore survives the failures an environment can produce, and records
that it failed instead of raising or silently omitting the entry:

| Probe | Failure it survives | What the record shows |
|---|---|---|
| `git` subprocesses | missing binary, timeout, non-zero exit, undecodable output | `sha`/`branch` null; `dirty: "unknown"` when `status` fails |
| dirty-tree fingerprint | a diff that is not valid UTF-8 (a modified tracked binary or non-UTF-8 text file) | hashed over git's raw bytes; `<probe failed: git diff>` if git fails |
| distribution versions | unreadable or malformed package metadata | `<probe failed: ...>` as the version |
| host and device facts | no passwd entry for the uid, no backend | `<probe failed: ...>` for that field only |
| input hashing | file readable by `stat` but not by `open` | size and mtime kept, `sha256_error` marker |

Only environmental exception types (`OSError`, `ValueError`, `RuntimeError`,
`KeyError`, `subprocess.SubprocessError`) are caught. A `TypeError` or
`AttributeError` is a bug in the probe and stays loud, and a data-mirror
revision changing mid-process (`RuntimeError` from `jcm.data.remote`) is a
real inconsistency in the run's inputs, not a probe failure, so it still
raises.

The dirty-tree fingerprint runs `git diff --no-ext-diff --no-textconv HEAD`,
so the bytes hashed are git's own and do not vary with a user's diff-driver
configuration.
