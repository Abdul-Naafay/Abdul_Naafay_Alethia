#!/usr/bin/env python3
"""Resumable driver for a small-disk machine (free Colab/Kaggle): download -> process -> delete.

    python colab_driver.py --parts /content/drive/MyDrive/vi_parts --workdir /content/fv

For every batch of N parquet files it downloads them (the next batch is prefetched while the
current one is processed), runs run.py with --resume, writes <parts>/part_XXX.jsonl, marks the
batch done, deletes the parquet files. Re-running the same command after a disconnect continues
where it stopped (finished batches are skipped, an unfinished one is resumed video by video).
Finally it merges all parts into results.jsonl.gz.

Needs: HF_TOKEN env var (read token, terms of the dataset accepted) unless --source-dir is given.
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

REPO = "HuggingFaceFV/finevideo"
HERE = os.path.dirname(os.path.abspath(__file__))


def fname(i):
    return "train-%05d-of-01357.parquet" % i


def fetch(batch_files, dest, source_dir, token):
    """Download (or copy) the given file indexes into dest/. Returns list of local paths."""
    os.makedirs(dest, exist_ok=True)
    out = []
    for i in batch_files:
        target = os.path.join(dest, fname(i))
        if os.path.exists(target) and not os.path.exists(target + ".part"):
            out.append(target)
            continue
        if source_dir:
            shutil.copy(os.path.join(source_dir, fname(i)), target)
        else:
            from huggingface_hub import hf_hub_download
            for attempt in range(5):
                try:
                    p = hf_hub_download(REPO, "data/" + fname(i), repo_type="dataset", token=token,
                                        local_dir=os.path.join(dest, "_hf"))
                    shutil.move(p, target)
                    break
                except Exception as e:  # network hiccup: retry
                    print("download of %s failed (%r), retry %d" % (fname(i), e, attempt + 1), flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError("could not download %s" % fname(i))
        out.append(target)
    shutil.rmtree(os.path.join(dest, "_hf"), ignore_errors=True)
    return out


class Prefetch:
    def __init__(self, fn, *a):
        self.err = None
        self.t = threading.Thread(target=self._run, args=(fn, a), daemon=True)
        self.t.start()

    def _run(self, fn, a):
        try:
            fn(*a)
        except BaseException as e:  # noqa
            self.err = e

    def wait(self):
        self.t.join()
        if self.err:
            raise self.err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=50, help="exclusive")
    ap.add_argument("--batch-files", type=int, default=5)
    ap.add_argument("--workdir", default="/content/fv")
    ap.add_argument("--parts", default="vi_parts", help="where per-batch results are kept (use Google Drive!)")
    ap.add_argument("--final", default="results.jsonl.gz")
    ap.add_argument("--source-dir", default=None, help="testing: copy files from here instead of downloading")
    ap.add_argument("run_args", nargs=argparse.REMAINDER, help="extra args for run.py, after --")
    args = ap.parse_args()
    extra = [a for a in args.run_args if a != "--"]
    token = os.environ.get("HF_TOKEN")
    if not token and not args.source_dir:
        sys.exit("set HF_TOKEN (read token) first")
    os.makedirs(args.parts, exist_ok=True)

    batches = []
    for b0 in range(args.start, args.end, args.batch_files):
        batches.append(list(range(b0, min(b0 + args.batch_files, args.end))))
    names = ["part_%03d" % idxs[0] for idxs in batches]
    bdir = lambda k: os.path.join(args.workdir, "b%03d" % batches[k][0])
    todo = [k for k in range(len(batches)) if not os.path.exists(os.path.join(args.parts, names[k] + ".done"))]
    print("batches: %d total, %d to do" % (len(batches), len(todo)), flush=True)

    nxt = None
    t_all = time.time()
    for pos, k in enumerate(todo):
        t0 = time.time()
        if nxt is not None:
            nxt.wait()
        else:
            fetch(batches[k], bdir(k), args.source_dir, token)
        nxt = None
        if pos + 1 < len(todo):
            k2 = todo[pos + 1]
            nxt = Prefetch(fetch, batches[k2], bdir(k2), args.source_dir, token)
        t_dl = time.time() - t0
        out = os.path.join(args.parts, names[k] + ".jsonl")
        cmd = [sys.executable, os.path.join(HERE, "run.py"), "--input", bdir(k), "--output", out, "--resume"] + extra
        print("[%s] files %s ready (wait %.0fs) -> %s" % (names[k], batches[k], t_dl, " ".join(cmd[-6:])), flush=True)
        t1 = time.time()
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit("run.py failed with exit code %d on batch %s" % (rc, names[k]))
        open(os.path.join(args.parts, names[k] + ".done"), "w").write("%.1fs\n" % (time.time() - t1))
        shutil.rmtree(bdir(k), ignore_errors=True)
        print("[%s] done in %.0fs (total elapsed %.0fs)" % (names[k], time.time() - t1, time.time() - t_all), flush=True)

    from merge_results import main as merge
    merge(args.parts, args.final)


if __name__ == "__main__":
    main()
