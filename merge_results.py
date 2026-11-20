#!/usr/bin/env python3
"""Merge per-batch outputs into one file and optionally gzip it.

    python merge_results.py out_parts results.jsonl.gz
Parts are concatenated in file-name order; duplicate video groups are dropped (first one wins).
"""
import glob
import gzip
import json
import os
import sys


def main(parts_dir, out):
    seen_complete = set()
    n_lines = 0
    opener = gzip.open if out.endswith(".gz") else open
    with opener(out, "wt") as fo:
        for p in sorted(glob.glob(os.path.join(parts_dir, "*.jsonl"))):
            cur, buf = None, []

            def flush():
                nonlocal n_lines
                if cur is not None and cur not in seen_complete:
                    seen_complete.add(cur)
                    fo.write("".join(buf))
                    n_lines += len(buf)

            with open(p) as f:
                for line in f:
                    if not line.endswith("\n"):
                        continue
                    vid = json.loads(line)["video"]
                    if vid != cur:
                        flush()
                        cur, buf = vid, []
                    buf.append(line)
            flush()
    print("wrote %s: %d lines, %d videos" % (out, n_lines, len(seen_complete)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
