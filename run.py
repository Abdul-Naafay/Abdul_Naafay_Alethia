#!/usr/bin/env python3
"""Video indexing at speed: per-second object presence + people count with YOLO11l.

    python run.py --input <folder of .parquet files> --output results.jsonl

Pipeline
    parquet reader thread --> N decoder processes (PyAV decode, frame sampling,
    BGR + rotation + Ultralytics-exact letterbox, written to shared memory)
    --> main process (GPU: batched FP16 YOLO11l + class-aware NMS, per-second
    aggregation, JSONL writer).

Light imports only at module level so that spawned decoder processes start fast.
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

OBJECT_NAMES = ["person", "car", "dog", "laptop", "cell phone"]
COCO_IDS = [0, 2, 16, 63, 67]
IMGSZ = 640
STRIDE = 32
PAD_VALUE = 114
CONF = 0.4
IOU = 0.7


# ----------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_core.py)
# ----------------------------------------------------------------------------
def letterbox_params(h, w, imgsz=IMGSZ, stride=STRIDE):
    """Same maths as ultralytics LetterBox(auto=True, center=True, scaleup=True)."""
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = imgsz - new_w, imgsz - new_h
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return new_w, new_h, int(top), int(bottom), int(left), int(right)


def letterbox(img):
    """BGR uint8 HxWx3 -> letterboxed BGR uint8 (rectangular, multiple of 32)."""
    h, w = img.shape[:2]
    new_w, new_h, top, bottom, left, right = letterbox_params(h, w)
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )


def summarize_second(samples):
    """samples: list of (mask_bits, people). Returns (objects list, people int).

    Object present if found in at least half of the frames; people = median,
    rounded down when halfway between two integers.
    """
    n = len(samples)
    if n == 0:
        return [], 0
    objs = []
    for k, name in enumerate(OBJECT_NAMES):
        c = 0
        for m, _ in samples:
            c += (m >> k) & 1
        if c * 2 >= n:
            objs.append(name)
    ppl = sorted(p for _, p in samples)
    if n % 2:
        med = ppl[n // 2]
    else:
        med = (ppl[n // 2 - 1] + ppl[n // 2]) // 2
    return objs, int(med)


def video_lines(vid, per_sec, last_sec, error=None):
    out = []
    for s in range(0, last_sec + 1):
        objs, ppl = summarize_second(per_sec.get(s, ()))
        out.append(json.dumps({"video": vid, "second": s, "objects": objs, "people": ppl}))
    if error:
        out.append(json.dumps({"video": vid, "error": str(error)[:200]}))
    return out


class Sampler:
    """Streaming frame selector.

    Each second is split into K equal buckets; from every non-empty bucket one
    frame is kept: the first frame at/after the bucket centre, or (if the bucket
    has none) its last frame. K == 0 keeps every frame (reference mode).
    Times are exact integers: t = tn / den seconds.
    """

    def __init__(self, k, den):
        self.k = k
        self.den = den
        self.cur = -1
        self.taken = False
        self.pending = None

    def push(self, tn, frame):
        """Returns list of (frame, second) to keep."""
        den, k = self.den, self.k
        if k <= 0:
            return [(frame, tn // den)]
        keep = []
        b = (tn * k) // den
        if b < self.cur:
            b = self.cur
        if b > self.cur:
            if not self.taken and self.pending is not None:
                keep.append(self.pending)
            self.cur, self.taken, self.pending = b, False, None
        if not self.taken:
            if tn * 2 * k >= (2 * b + 1) * den:
                keep.append((frame, b // k))
                self.taken, self.pending = True, None
            else:
                self.pending = (frame, b // k)
        return keep

    def flush(self):
        if not self.taken and self.pending is not None:
            p, self.pending = self.pending, None
            return [p]
        return []


# ----------------------------------------------------------------------------
# Decoder worker
# ----------------------------------------------------------------------------
def get_rotation_cw(stream, frame=None):
    """Clockwise rotation (0/90/180/270) needed to display the video upright.

    Uses the display-matrix rotation exposed by PyAV on decoded frames (ccw degrees,
    same convention as ffprobe) and falls back to a legacy "rotate" metadata tag.
    """
    try:
        r = getattr(frame, "rotation", None) if frame is not None else None
        if r:
            return int(round(-float(r) / 90.0) * 90) % 360
    except Exception:
        pass
    try:
        tag = stream.metadata.get("rotate")
        if tag is not None:
            return int(round(float(tag) / 90.0) * 90) % 360
    except Exception:
        pass
    return 0


def rotate_cw(img, deg):
    if deg == 90:
        return np.ascontiguousarray(np.rot90(img, k=-1))
    if deg == 180:
        return np.ascontiguousarray(np.rot90(img, k=2))
    if deg == 270:
        return np.ascontiguousarray(np.rot90(img, k=1))
    return img


class ChunkWriter:
    """Packs letterboxed frames of one video into shared-memory slots."""

    def __init__(self, wid, gen, vid_idx, cap, slots, free_q, result_q):
        self.wid, self.gen, self.vid_idx, self.cap = wid, gen, vid_idx, cap
        self.slots, self.free_q, self.result_q = slots, free_q, result_q
        self.slot = None
        self.n = 0
        self.shape = None
        self.view = None
        self.secs = []

    def add(self, img, sec):
        h, w = img.shape[:2]
        if self.shape is not None and self.shape != (h, w):
            self.flush()
        if self.slot is None:
            self.slot = self.free_q.get()
            self.shape = (h, w)
            self.view = self.slots[self.slot, : self.cap * h * w * 3].reshape(self.cap, h, w, 3)
            self.n, self.secs = 0, []
        self.view[self.n] = img
        self.secs.append(sec)
        self.n += 1
        if self.n == self.cap:
            self.flush()

    def flush(self):
        if self.slot is None:
            return
        h, w = self.shape
        self.result_q.put(("chunk", self.wid, self.gen, self.vid_idx, self.slot, self.n, h, w, self.secs))
        self.slot, self.shape, self.view, self.n, self.secs = None, None, None, 0, []


def decode_video(data, k, threads, writer):
    """Decode `data`, push sampled frames to writer. Returns (last_sec, error)."""
    import av

    container = av.open(io.BytesIO(data))
    try:
        if not container.streams.video:
            return -1, "no video stream"
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
            stream.thread_count = threads
        except Exception:
            pass
        tb = stream.time_base
        num, den = tb.numerator, tb.denominator
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        rot = None  # decided on the first decoded frame
        sampler = Sampler(k, den)
        first = None
        last_sec = -1
        idx = 0
        err = None

        def take(items):
            for fr, sec in items:
                img = fr.to_ndarray(format="bgr24")
                if rot:
                    img = rotate_cw(img, rot)
                writer.add(letterbox(img), sec)

        try:
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.error.FFmpegError:
                    continue  # damaged packet: skip it, keep going
                for frame in frames:
                    pts = frame.pts
                    if pts is None:
                        pts = int(round(idx / fps * den / num))
                    idx += 1
                    if first is None:
                        first = pts
                        rot = get_rotation_cw(stream, frame)
                    tn = max(0, (pts - first) * num)
                    sec = tn // den
                    if sec > last_sec:
                        last_sec = sec
                    take(sampler.push(tn, frame))
            take(sampler.flush())
        except av.error.FFmpegError as e:  # truncated / broken container mid-way
            try:
                take(sampler.flush())
            except Exception:
                pass
            err = "decode failed: %s" % e
        if idx == 0:
            return -1, err or "could not decode video"
        return last_sec, err
    finally:
        container.close()


def worker_main(wid, gen, task_q, result_q, free_q, shm_name, nslots, slot_bytes, cap, k, threads):
    """Decoder process. Owns its own slot pool (free_q) so that killing one worker can never
    leave a shared lock held and block the others."""
    cv2.setNumThreads(1)
    shm = shared_memory.SharedMemory(name=shm_name)
    slots = np.ndarray((nslots, slot_bytes), dtype=np.uint8, buffer=shm.buf)
    while True:
        task = task_q.get()
        if task is None:
            break
        vid_idx, data = task
        result_q.put(("start", wid, gen, vid_idx))
        writer = ChunkWriter(wid, gen, vid_idx, cap, slots, free_q, result_q)
        last_sec, err = -1, None
        try:
            last_sec, err = decode_video(data, k, threads, writer)
        except Exception as e:  # any failure -> error line, worker lives on
            err = "%s: %s" % (type(e).__name__, e)
        try:
            writer.flush()
        except Exception:
            pass
        del data
        result_q.put(("done", wid, gen, vid_idx, last_sec, err))
    shm.close()


# ----------------------------------------------------------------------------
# GPU detector
# ----------------------------------------------------------------------------
class Detector:
    def __init__(self, weights, device, half):
        import torch
        from ultralytics import YOLO

        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:  # older ultralytics
            from ultralytics.utils.ops import non_max_suppression
        self.nms = non_max_suppression
        self.torch = torch
        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"
        net = YOLO(weights).model
        names = net.names
        got = [names[i] for i in COCO_IDS]
        if got != OBJECT_NAMES:
            raise RuntimeError("unexpected class names %s" % got)
        net = net.fuse(verbose=False) if hasattr(net, "fuse") else net
        net = net.to(self.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        net = net.half() if self.half else net.float()
        self.net = net
        lut = torch.full((len(names),), -1, dtype=torch.long)
        for j, c in enumerate(COCO_IDS):
            lut[c] = j
        self.lut = lut.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def _is_oom(e):
        return "out of memory" in str(e).lower()

    def counts(self, frames_bgr):
        """frames_bgr: uint8 tensor (B,H,W,3) on device, BGR. -> np.int64 (B,5)."""
        try:
            return self._counts(frames_bgr)
        except RuntimeError as e:
            if not self._is_oom(e) or frames_bgr.shape[0] == 1:
                raise
            self.torch.cuda.empty_cache()
            mid = frames_bgr.shape[0] // 2
            return np.concatenate([self.counts(frames_bgr[:mid]), self.counts(frames_bgr[mid:])])

    def _counts(self, frames_bgr):
        torch = self.torch
        with torch.inference_mode():
            x = frames_bgr.flip(-1).permute(0, 3, 1, 2).contiguous()  # BGR->RGB, NCHW
            x = (x.half() if self.half else x.float()).div_(255)
            preds = self.net(x)
            dets = self.nms(
                preds, CONF, IOU, classes=COCO_IDS, agnostic=False, max_det=300,
                max_time_img=1e6,  # never truncate NMS on slow devices
            )
            b = len(dets)
            idx = torch.cat([torch.full((d.shape[0],), i, dtype=torch.long, device=x.device)
                             for i, d in enumerate(dets)])
            cls = torch.cat([d[:, 5] for d in dets]).long()
            m = self.lut[cls]
            keep = m >= 0
            flat = idx[keep] * 5 + m[keep]
            return torch.bincount(flat, minlength=b * 5).view(b, 5).cpu().numpy()

    def warmup(self, h=384, w=640, b=2):
        torch = self.torch
        dummy = torch.zeros((b, h, w, 3), dtype=torch.uint8, device=self.device)
        self.counts(dummy)


# ----------------------------------------------------------------------------
# Output file handling
# ----------------------------------------------------------------------------
def prepare_output(path, resume):
    """Open output. With resume, keep complete videos, drop a possibly partial last one."""
    done = set()
    if resume and os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read()
        end = raw.rfind(b"\n") + 1  # drop a torn last line
        raw = raw[:end]
        # group start offsets: the last video group may be incomplete -> redo it
        pos, last_vid, last_start = 0, None, 0
        for line in raw.split(b"\n"):
            if not line:
                pos += 1
                continue
            try:
                vid = json.loads(line)["video"]
            except Exception:
                vid = None
            if vid != last_vid:
                last_vid, last_start = vid, pos
            if vid is not None:
                done.add(vid)
            pos += len(line) + 1
        if last_vid is not None:
            done.discard(last_vid)
            raw = raw[:last_start]
        with open(path, "wb") as f:
            f.write(raw)
        return open(path, "a", buffering=1 << 20), done
    return open(path, "w", buffering=1 << 20), done


class Writer:
    def __init__(self, f):
        self.f = f
        self.lock = threading.Lock()
        self.videos = 0
        self.errors = 0

    def write(self, lines, is_error=False):
        with self.lock:
            self.f.write("\n".join(lines) + "\n")
            self.f.flush()
            self.videos += 1
            self.errors += int(is_error)


# ----------------------------------------------------------------------------
# Reader thread
# ----------------------------------------------------------------------------
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.names = {}        # vid_idx -> video id
        self.submitted = 0
        self.reader_done = False


def reader_thread(files, done, task_q, writer, shared, limit):
    import pyarrow.parquet as pq

    count = 0

    def err_lines(vid, msg):
        if vid in done:  # resume: this error line was already written
            return
        writer.write([json.dumps({"video": vid, "error": msg[:200]})], is_error=True)

    try:
        for path in files:
            name = os.path.basename(path)
            try:
                pf = pq.ParquetFile(path)
                if "mp4" not in pf.schema_arrow.names:
                    raise ValueError("no mp4 column")
                n_rg = pf.num_row_groups
                sizes = [pf.metadata.row_group(i).num_rows for i in range(n_rg)]
            except Exception as e:
                err_lines("%s:0" % name, "could not read parquet file: %s" % e)
                continue
            row0 = 0
            for rg in range(n_rg):
                nrows = sizes[rg]
                ids = ["%s:%d" % (name, row0 + i) for i in range(nrows)]
                if all(v in done for v in ids):
                    row0 += nrows
                    continue
                try:
                    col = pf.read_row_group(rg, columns=["mp4"]).column(0)
                except Exception as e:
                    for v in ids:
                        if v not in done:
                            err_lines(v, "could not read parquet row group: %s" % e)
                    row0 += nrows
                    continue
                for i in range(nrows):
                    vid = ids[i]
                    if vid in done:
                        continue
                    if limit and count >= limit:
                        return
                    count += 1
                    try:
                        data = col[i].as_py()
                    except Exception as e:
                        err_lines(vid, "could not read row: %s" % e)
                        continue
                    if not data:
                        err_lines(vid, "empty video")
                        continue
                    with shared.lock:
                        idx = shared.submitted
                        shared.names[idx] = vid
                        shared.submitted += 1
                    task_q.put((idx, data))  # blocks when decoders are saturated
                    del data
                del col
                row0 += nrows
    except Exception as e:  # never let the reader die silently
        print("reader error: %r" % e, file=sys.stderr)
    finally:
        with shared.lock:
            shared.reader_done = True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with .parquet files (searched recursively) or one file")
    ap.add_argument("--output", required=True, help="output .jsonl")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--weights", default=os.path.join(here, "yolo11l.pt"))
    ap.add_argument("--samples", type=int, default=5,
                    help="frames analysed per second of video (0 = every frame = reference mode)")
    ap.add_argument("--batch", type=int, default=32, help="max frames per GPU batch")
    ap.add_argument("--workers", type=int, default=0, help="decoder processes (0 = auto)")
    ap.add_argument("--device", default="cuda:0" if _cuda() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="disable FP16")
    ap.add_argument("--resume", action="store_true", help="append to an existing output, skipping finished videos")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N videos (testing)")
    ap.add_argument("--video-timeout", type=float, default=900, help="kill a decoder stuck on one video (s)")
    return ap.parse_args(argv)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def find_parquets(path):
    if os.path.isfile(path):
        return [path]
    files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
    return sorted(files)


def main(argv=None):
    args = parse_args(argv)
    files = find_parquets(args.input)
    if not files:
        print("no .parquet files found in %s" % args.input, file=sys.stderr)
        return 2
    t0 = time.time()
    ncpu = os.cpu_count() or 2
    nworkers = args.workers or max(1, min(ncpu, 16))
    threads = max(1, ncpu // nworkers)
    cap = max(1, args.batch)
    slot_bytes = cap * IMGSZ * IMGSZ * 3
    per_worker_slots = 3
    total_slots = nworkers * per_worker_slots

    out_f, done = prepare_output(args.output, args.resume)
    writer = Writer(out_f)
    if done:
        print("resume: %d videos already done" % len(done), file=sys.stderr)

    # decoder processes are started before CUDA is initialised
    ctx = mp.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=total_slots * slot_bytes)
    slots = np.ndarray((total_slots, slot_bytes), dtype=np.uint8, buffer=shm.buf)
    task_q = ctx.Queue(maxsize=nworkers * 2)
    result_q = ctx.Queue()
    free_qs = {}
    gens = {}
    procs = {}

    def spawn(wid):
        """(Re)start decoder `wid` with a fresh slot queue and a new generation number."""
        gens[wid] = gens.get(wid, -1) + 1
        fq = ctx.Queue()
        for s_ in range(wid * per_worker_slots, (wid + 1) * per_worker_slots):
            fq.put(s_)
        free_qs[wid] = fq
        p = ctx.Process(
            target=worker_main, daemon=True,
            args=(wid, gens[wid], task_q, result_q, fq, shm.name, total_slots, slot_bytes, cap, args.samples, threads),
        )
        p.start()
        procs[wid] = p

    for w in range(nworkers):
        spawn(w)

    shared = Shared()
    rt = threading.Thread(target=reader_thread, args=(files, done, task_q, writer, shared, args.limit), daemon=True)
    rt.start()

    if not args.device.startswith("cuda"):
        print("WARNING: running on %s - this will be extremely slow; is CUDA available?" % args.device, file=sys.stderr)
    det = Detector(args.weights, args.device, not args.fp32)
    det.warmup()
    torch = det.torch
    print("ready: device=%s fp16=%s workers=%d samples/s=%d batch=%d (%.1fs setup)" % (
        det.device, det.half, nworkers, args.samples, cap, time.time() - t0), file=sys.stderr)

    acc = {}            # vid_idx -> dict(per_sec, last_seen)
    current = {}        # wid -> (vid_idx, start_time)
    completed = set()
    frames_done = 0
    t_run = time.time()
    last_report = t_run
    last_msg = t_run

    def finalize(idx, last_sec, err):
        a = acc.pop(idx, None)
        if idx in completed:
            return
        completed.add(idx)
        vid = shared.names[idx]
        per_sec = a["per_sec"] if a else {}
        if last_sec < 0 and per_sec:
            last_sec = max(per_sec)
        if last_sec < 0:
            writer.write([json.dumps({"video": vid, "error": (err or "could not decode video")[:200]})], True)
        else:
            writer.write(video_lines(vid, per_sec, last_sec, err), bool(err))

    try:
        while True:
            try:
                msg = result_q.get(timeout=2)
            except queue.Empty:
                msg = None
            now = time.time()
            if msg is not None:
                last_msg = now
                kind, wid, gen = msg[0], msg[1], msg[2]
                if gen != gens.get(wid):
                    continue  # leftovers of a decoder we already replaced
                if kind == "start":
                    idx = msg[3]
                    current[wid] = (idx, now)
                    acc.setdefault(idx, {"per_sec": {}})
                elif kind == "chunk":
                    _, _, _, idx, slot, n, h, w, secs = msg
                    a = acc.get(idx)
                    if a is None or idx in completed:
                        free_qs[wid].put(slot)
                        continue
                    current[wid] = (idx, now)  # heartbeat for the stuck-decoder watchdog
                    view = slots[slot, : n * h * w * 3].reshape(n, h, w, 3)
                    t = torch.from_numpy(view)
                    t = t.to(det.device) if det.device.type != "cpu" else t.clone()
                    free_qs[wid].put(slot)
                    try:
                        cnt = det.counts(t)
                        ps = a["per_sec"]
                        for j in range(n):
                            row = cnt[j]
                            mask = 0
                            for kk in range(5):
                                if row[kk] > 0:
                                    mask |= 1 << kk
                            ps.setdefault(secs[j], []).append((mask, int(row[0])))
                        frames_done += n
                    except Exception as e:
                        print("inference error on %s: %r" % (shared.names.get(idx), e), file=sys.stderr)
                        a["error"] = "inference failed: %s" % e
                elif kind == "done":
                    _, _, _, idx, last_sec, err = msg
                    current.pop(wid, None)
                    a = acc.get(idx)
                    if a is not None and a.get("error") and not err:
                        err = a["error"]
                    finalize(idx, last_sec, err)

            # supervise decoders: crashed or stuck processes
            for wid, p in list(procs.items()):
                stuck = wid in current and now - current[wid][1] > args.video_timeout
                if p.is_alive() and not stuck:
                    continue
                if stuck and p.is_alive():
                    p.terminate()
                    p.join(5)
                if wid in current:
                    idx = current.pop(wid)[0]
                    reason = "decoder timed out" if stuck else "decoder crashed"
                    a = acc.get(idx)
                    finalize(idx, max(a["per_sec"]) if a and a["per_sec"] else -1, reason)
                spawn(wid)

            with shared.lock:
                rd, sub = shared.reader_done, shared.submitted
            if rd and len(completed) >= sub:
                break
            # safety net: a task lost between queue.get() and the "start" message
            if rd and now - last_msg > 120 and not current and len(completed) < sub:
                for idx in range(sub):
                    if idx not in completed:
                        finalize(idx, -1, "video lost (decoder died)")
                break

            if now - last_report > 15:
                last_report = now
                el = now - t_run
                print("[%6.0fs] videos done %d/%s  frames %d  %.1f fps" % (
                    el, len(completed), "?" if not rd else sub, frames_done, frames_done / max(el, 1e-9)),
                    file=sys.stderr)
    finally:
        for _ in procs:
            try:
                task_q.put_nowait(None)
            except Exception:
                pass
        for p in procs.values():
            p.join(3)
            if p.is_alive():
                p.terminate()
        writer.f.close()
        del slots
        shm.close()
        shm.unlink()

    el = time.time() - t_run
    print("finished: %d videos (%d with errors), %d frames inferred, %.1fs (%.1f frames/s)" % (
        writer.videos, writer.errors, frames_done, el, frames_done / max(el, 1e-9)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
