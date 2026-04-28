"""Compare a speed_test.py result against a baseline JSON.

Exits 0 if ``min_s <= CEILING_FACTOR * baseline.min_s``, non-zero otherwise.

Usage::

    python speed_test.py > result.json
    python compare_benchmark.py result.json benchmark_results/jcm-bench-cpu-c4.json

The baseline JSON is the committed dev-branch reference measured on the same
hardware (see ``run_benchmarks.sh``). Run the comparison on matching hardware
or the result is meaningless.
"""

import argparse
import json
import sys

CEILING_FACTOR = 1.2


def load_result(path):
    # speed_test.py prepends human-readable lines; the final JSON blob is the
    # last {...} in the file. Find it by scanning from the last '{'.
    text = open(path).read()
    start = text.rfind("{\n")
    if start < 0:
        raise ValueError(f"No JSON object found in {path}")
    return json.loads(text[start:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result")
    ap.add_argument("baseline")
    ap.add_argument("--ceiling-factor", type=float, default=CEILING_FACTOR)
    args = ap.parse_args()

    result = load_result(args.result)
    baseline = load_result(args.baseline)

    # Sanity-check hardware match — different device_kind makes comparison invalid.
    for k in ("platform", "device_kind"):
        if result.get(k) != baseline.get(k):
            print(
                f"WARNING: {k} differs (result={result.get(k)!r}, "
                f"baseline={baseline.get(k)!r}) — comparison not meaningful.",
                file=sys.stderr,
            )

    ceiling = args.ceiling_factor * baseline["min_s"]
    passed = result["min_s"] <= ceiling

    print(f"baseline min_s = {baseline['min_s']:.3f}s")
    print(f"result   min_s = {result['min_s']:.3f}s")
    print(f"ceiling  ({args.ceiling_factor}x) = {ceiling:.3f}s")
    print("PASS" if passed else "FAIL: regression exceeds ceiling")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
