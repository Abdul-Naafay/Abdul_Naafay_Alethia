# Video indexing at speed (YOLO11l, FineVideo)

For each second of each video: which of `person, car, dog, laptop, cell phone` appear, and how many
people are visible. Output is JSON Lines as specified in the brief.

## Install (grader machine: Ubuntu 22.04, Python 3.11, CUDA 12 driver)

```bash
sudo apt-get install -y libgl1 libglib2.0-0
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (needs internet, run once before the timed run)

```bash
python prepare.py
```

Downloads `yolo11l.pt` next to `run.py` and checks that it loads. The timed run needs no internet.

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

Useful options: `--samples N` (frames analysed per second, default 5; `0` = every frame, the
reference mode), `--batch 32`, `--workers N` (decoder processes, default = CPU threads up to 16),
`--fp32`, `--resume`, `--limit N`. The default is the speed/accuracy setting described below.

## How it works

1. A reader thread streams the `mp4` column of every Parquet file row group by row group.
2. Decoder processes (PyAV, software H.264/etc.) decode each video, give every frame an exact
   integer second (`floor(pts - first_pts)`), keep `--samples` frames per second, convert only
   those to BGR, apply the rotation tag, and letterbox them exactly like Ultralytics does
   (rectangular minimum padding, multiple of 32, pad value 114). They write the frames into shared
   memory, so no big arrays are pickled.
3. The main process batches frames onto the GPU, runs YOLO11l (FP16 on CUDA) and Ultralytics
   class-aware NMS (conf 0.4, IoU 0.7), and counts the five classes per frame on the GPU.
4. Per second: an object is present if it is in at least half of the analysed frames; people =
   median (rounded down at .5). One line per second up to the last frame's second.
5. A damaged video gives an `error` line (lines already produced for it are kept). A crashed or
   stuck decoder is killed and replaced; a damaged parquet file or row gives error lines too.

## Resumable run on small disks (free Colab / Kaggle)

```bash
pip install ultralytics==8.4.215 av==18.1.0 pyarrow==26.0.1 huggingface_hub   # keep Colab's own torch
python prepare.py
export HF_TOKEN=hf_xxx            # your read token (do NOT commit it)
python colab_driver.py --end 50 --batch-files 5 --workdir /content/fv \
       --parts /content/drive/MyDrive/vi_parts -- --samples 5
```

It downloads 5 files at a time (prefetching the next batch), processes, deletes them, and keeps
one result part per batch on Google Drive. After a disconnect, run the same command again.
At the end it writes `results.jsonl.gz`.

## Choosing `--samples` (check correctness before the full run)

```bash
python run.py --input small_folder --output ref.jsonl --samples 0     # every frame = reference
python run.py --input small_folder --output k3.jsonl  --samples 3
python score.py ref.jsonl k3.jsonl                                      # prints the 3 thresholds
```

Use about 20-30 real FineVideo videos. Pick the smallest `--samples` that passes with a margin
(odd numbers give a cleaner median for the people count). Also compare `--fp32` and FP16 once.

## Tests

```bash
python tests/test_core.py                       # letterbox == Ultralytics, sampling, aggregation
python tests/test_detector_equivalence.py       # batched pipeline == ultralytics predict(), per frame
python tests/make_test_parquet.py <dir_with_a.mp4,b.mp4,rot_dm.mp4> pq_test   # broken-input smoke test
```
