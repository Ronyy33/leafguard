"""Turn pytest's JUnit XML into a DVC metrics file + a per-test table.

Usage: python scripts/test_report.py reports/tests/junit.xml
Writes: reports/test_summary.json, reports/tests/test_cases.md
Exits non-zero if any test failed, so the DVC `test` stage fails the CI pipeline.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

root = ET.parse(sys.argv[1]).getroot()
suite = root if root.tag == "testsuite" else root.find("testsuite")
rows, counts = [], {"passed": 0, "failed": 0, "skipped": 0}
for tc in suite.iter("testcase"):
    if tc.find("failure") is not None or tc.find("error") is not None:
        status = "failed"
    elif tc.find("skipped") is not None:
        status = "skipped"
    else:
        status = "passed"
    counts[status] += 1
    rows.append((tc.get("classname", "").replace("tests.", ""), tc.get("name"), status, float(tc.get("time", 0))))

summary = {
    "tests_total": len(rows),
    **{f"tests_{k}": v for k, v in counts.items()},
    "tests_pass_rate": round(counts["passed"] / max(1, len(rows) - counts["skipped"]), 4),
    "tests_seconds": round(float(suite.get("time", 0)), 2),
}
Path("reports/test_summary.json").write_text(json.dumps(summary, indent=2))

md = ["| # | Module | Test | Result | Time (s) |", "|---|---|---|---|---|"]
md += [
    f"| {i} | {m} | `{n}` | {'✅ pass' if s == 'passed' else '⏭ skip' if s == 'skipped' else '❌ FAIL'} | {t:.2f} |"
    for i, (m, n, s, t) in enumerate(rows, 1)
]
Path("reports/tests/test_cases.md").write_text("\n".join(md) + "\n")
print(json.dumps(summary))
sys.exit(1 if counts["failed"] else 0)
