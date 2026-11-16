#!/usr/bin/env python
"""Pool evaluation folders (any number of evaluation seeds and parallel pieces) and print success rates,
95% intervals and paired comparisons.

Example
  python scripts/summarize.py results/seed720000 results/seed750000 results/seed760000
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifier import rollout as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path, help="folders that contain episodes.json at any depth")
    ap.add_argument("--json", type=Path, default=None, help="also write the summary to this file")
    a = ap.parse_args()
    rows = [r for folder in a.folders for f in sorted(folder.rglob("episodes.json")) for r in json.loads(f.read_text())]
    if not rows:
        raise SystemExit("no episodes found")
    summary = R.summarize(rows)
    by_suite = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        suite = r["task"].rsplit("_task", 1)[0]
        by_suite[suite][r["mode"]] += bool(r["success"])
        by_suite[suite][r["mode"] + "#"] += 1
    print("Condition            episodes  success rate  95% interval")
    for mode, c in summary["conditions"].items():
        print(f"{mode:20s} {c['episodes']:8d}  {100 * c['success_rate']:11.1f}%  [{100 * c['wilson95'][0]:.1f}, {100 * c['wilson95'][1]:.1f}]")
    print("\nSuccess rate by suite")
    for suite in sorted(by_suite):
        print(f"{suite:16s} " + "  ".join(f"{m}: {100 * by_suite[suite][m] / by_suite[suite][m + '#']:.1f}%" for m in summary["conditions"]))
    print("\nPaired comparisons (same task and episode seed)")
    for name, p in summary["paired"].items():
        print(f"{name}: {100 * p['difference']:+.1f} pp [{100 * p['bootstrap95'][0]:+.1f}, {100 * p['bootstrap95'][1]:+.1f}], "
              f"solved by only the first / only the second: {100 * p['only_first'] / p['paired_episodes']:.1f}% / {100 * p['only_second'] / p['paired_episodes']:.1f}% of {p['paired_episodes']} paired episodes, "
              f"exact McNemar p = {p['mcnemar_exact_p']:.2g}")
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
