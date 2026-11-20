#!/usr/bin/env python3
"""Preparation step (has internet, not timed): download yolo11l.pt next to run.py."""
import os
import sys
import urllib.request

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11l.pt")
MIN_SIZE = 40_000_000  # the real file is ~51 MB


def ok():
    return os.path.exists(DEST) and os.path.getsize(DEST) > MIN_SIZE


if not ok():
    print("downloading", URL)
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as e:  # fall back to the Ultralytics downloader
        print("direct download failed (%r), trying ultralytics" % e)
        from ultralytics.utils.downloads import attempt_download_asset
        p = attempt_download_asset("yolo11l.pt")
        if os.path.abspath(str(p)) != DEST:
            os.replace(str(p), DEST)
if not ok():
    sys.exit("yolo11l.pt missing or incomplete")

# load once to be sure the weights and the installed ultralytics/torch work together
from ultralytics import YOLO  # noqa: E402
m = YOLO(DEST)
print("weights OK, classes:", [m.model.names[i] for i in (0, 2, 16, 63, 67)])
import torch  # noqa: E402
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available; run.py would fall back to the (very slow) CPU")
