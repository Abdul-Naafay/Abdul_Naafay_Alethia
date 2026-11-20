#!/usr/bin/env python3
"""Score a candidate results file against a reference using the brief's three thresholds.

    python score.py reference.jsonl candidate.jsonl

Reference = run.py --samples 0 (every frame) on the same videos.
Only seconds present in the reference are scored; a second missing from the candidate counts as
"no objects, 0 people". Videos where the reference has an error line are skipped.
"""
import json
import sys

NAMES = ["person", "car", "dog", "laptop", "cell phone"]


def load(path):
    secs, errs = {}, set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                errs.add(d["video"])
            else:
                secs[(d["video"], d["second"])] = (frozenset(d["objects"]), d["people"])
    return secs, errs


def main(ref_path, cand_path):
    ref, ref_err = load(ref_path)
    cand, cand_err = load(cand_path)
    tp = {n: 0 for n in NAMES}
    fp = dict(tp)
    fn = dict(tp)
    exact = within1 = total = missing = 0
    for key, (robjs, rppl) in ref.items():
        if key[0] in ref_err:
            continue
        total += 1
        if key in cand:
            cobjs, cppl = cand[key]
        else:
            cobjs, cppl, missing = frozenset(), 0, missing + 1
        for n in NAMES:
            r, c = n in robjs, n in cobjs
            tp[n] += r and c
            fp[n] += (not r) and c
            fn[n] += r and (not c)
        exact += cppl == rppl
        within1 += abs(cppl - rppl) <= 1
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    f1 = 1.0 if (2 * TP + FP + FN) == 0 else 2 * TP / (2 * TP + FP + FN)
    print("seconds scored      : %d (missing in candidate: %d)" % (total, missing))
    for n in NAMES:
        d = 2 * tp[n] + fp[n] + fn[n]
        print("  %-10s F1 %s  (tp %d fp %d fn %d)" % (n, "%.4f" % (2 * tp[n] / d) if d else "  n/a ", tp[n], fp[n], fn[n]))
    pe = exact / max(total, 1)
    pw = within1 / max(total, 1)
    ok = (f1 >= 0.95, pe >= 0.85, pw >= 0.95)
    print("objects micro-F1    : %.4f   (need >= 0.95)  %s" % (f1, "PASS" if ok[0] else "FAIL"))
    print("people exact        : %.4f   (need >= 0.85)  %s" % (pe, "PASS" if ok[1] else "FAIL"))
    print("people within 1     : %.4f   (need >= 0.95)  %s" % (pw, "PASS" if ok[2] else "FAIL"))
    extra = [v for v in cand_err if v not in ref_err]
    if extra:
        print("note: %d videos have errors only in the candidate, e.g. %s" % (len(extra), extra[:3]))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
