# Write-up


## 1. Your timings (max 100 words)
On a 20 video test, running every frame took 1412 secs (basically avg of 115 FPS which would be around 31 hrs for all 50 files), while the final version with --samples 3 took 210 secs which is about 6.7× (:>) faster. The full 50 file run finished in 4 hours and 16 minutes. Sampling 3 evenly spaced frames per second instead of every frame saved the most time.

## 2. Your pipeline (max 150 words)
I run a reader thread that pulls rows from each Parquet file using PyArrow and drops them into a queue. Worker processes then take the video bytes, decode them with PyAV, rotate frames if the stream has a rotation tag and select up to three evenly spaced frames each second. Each frame is letterboxed to 640 by 640 pixels in BGR format to match Ultralytics and then stored in shared memory. If worker processes hang or crash, I restart worker processes and broken videos produce an error line. The run keeps going. The main process then pulls batches from shared memory, runs FP16 YOLO11l with class‑aware NMS (confidence >= 0.4, iou = 0.7), counts detections for the five classes, takes the majority presence and median people count and writes JSONL.

## 3. Where the time goes (max 150 words)
Video decoding with PyAV and letterbox resizing on the CPU take the time. On the Colab machine, I only had two vCPUs feeding a Tesla T4 GPU. The GPU then spent time waiting for decoded frames of running at full capacity. Across the 50 file run, the ten batches of about 165 videos averaged 1533s each, totaling 15327s. This bottleneck was caused by video decoding of 125 hours of video on two CPU cores. I measured this using stage timers in run.py and batch wall clock logs in colab_driver.py across both the 20 video benchmark and the full 50 file run.

## 4. Checking correctness (max 100 words)
First I ran unit tests to make sure my letterbox math was the same as Ultralytics. I also checked that 30 frames of custom FP16 batched inference gave the detections as predict function. Then I ran all frames with --samples 0 on 20 videos, totaling 5763 secs. I used score.py to score --samples 3 against this ground truth. I found that objects reached an F1 score of 0.9834. The exact people count matched 95.19% of the secs. The within-1 count matched 99.81% of the secs. In the end I ran a format check across all 461757 output secs.

## 5. Working with AI (max 100 words)
An AI assistant at first went to --samples 5 and added opencv-python==5.0.0.93 in requirements.txt. When I benchmarked both sampling rates on 20 real videos, --samples 3 was 28% faster while still scoring 0.9834 F1 and 95.19% exact people accuracy therefore I switched the default to 3. I also dropped the OpenCV 5.0 pin because that version doesnt exist on PyPI.

## 6. Next steps (max 100 words)
If I had one more week, I would make two changes to the RTX 4090 grading machine. First, I would shift video decoding from the CPU to hardware NVDEC. I would use PyAV/FFmpeg with CUDA support for this. CPU decoding was my slowdown. Second, I would export yolo11l.pt to an FP16 TensorRT engine inside prepare.py. That step runs untimed on the target GPU, so it would make inference faster. Then I would test the install and run on a Python 3.11 + CUDA 12 setup.


