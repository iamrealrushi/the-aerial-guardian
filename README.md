# The Aerial Guardian: Person Detection in Drone Videos

**EE5340: Video Analytics — Mini-Project | IIT Guwahati**  
*Solving small-object and motion-blurred person detection in aerial drone footage.*

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://iamrealrushi.github.io/the-aerial-guardian/)
[![Dataset](https://img.shields.io/badge/Dataset-VisDrone2019-green)](https://github.com/VisDrone/VisDrone-Dataset)
[![Model](https://img.shields.io/badge/Model-YOLOv8s--P2-orange)](https://github.com/ultralytics/ultralytics)

---

## Overview

Detecting people from drones is difficult because human targets are tiny—in VisDrone-DET validation, **89% of person boxes are smaller than 32×32 px** (median side: **17 px**)—and moving UAV cameras add motion blur.

This project builds a frame-by-frame **person-only detector** (no tracking) trainable on a single free **Google Colab T4 GPU**, tuned specifically for **COCO `AP_small`**.

### Key Ideas
- **YOLOv8s-P2 (Stride-4 Head):** Adds a high-resolution P2 detection head (`192×192` at `768 px` input) to preserve tiny pedestrian details.
- **Train–Test Matched Tiling:** Slices frames into `512×512` tiles (25% overlap) and upscales by **1.5× to `768×768`** during both training and inference, plus a `1280 px` full-frame pass for large foreground persons.
- **Hybrid CIoU + NWD Loss:** Blends CIoU with Normalized Wasserstein Distance (`0.5·CIoU + 0.5·NWD`, `C = 12.8 px`) for smoother gradients on tiny boxes.
- **Drone Augmentations & Copy-Paste:** Offline Gaussian-feathered box-level copy-paste of small persons (`p = 0.30`) + online motion blur, defocus, sensor noise, exposure jitter, and JPEG compression.
- **Memory-Safe Training:** Image-by-image chunked `TaskAlignedAssigner` prevents GPU OOM crashes on crowded tiles[cite: 38].
- **Blur-Aware Video Demo:** Uses Laplacian variance to lower the confidence threshold on blurry frames (`0.20` vs. `0.30` on sharp frames)[cite: 38].

---

## Results

Evaluated with official COCO metrics (`pycocotools`) after merging VisDrone `pedestrian` + `people` → `person` and discarding ignored regions[cite: 38]:

| Setting | maxDets | AP (%) | AP50 (%) | AP75 (%) | AP_small (%) | AP_medium (%) | AP_large (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DET val — Sliced (Ours)** | 100 | **33.6** | **70.6** | **26.4** | **31.1** | **49.8** | 71.6 |
| DET val — Full-frame @1280 (Baseline) | 100 | 27.7 | 62.2 | 19.3 | 24.5 | 48.1 | 72.5 |
| **DET val — Sliced (Crowded)** | 500 | — | **73.7** | **27.1** | **32.5** | **50.6** | 71.6 |
| **VID val — Sliced (Ours)** | 100 | 25.5 | 62.4 | 16.1 | 7.0 | 31.1 | 40.8 |

- **+6.6 `AP_small` (+27% relative)** over full-frame `@1280` inference with the same weights[cite: 38].
- **Runtime (Tesla T4, FP16):** `0.14 s/image` (~7.1 FPS) on DET val; `0.26 s/frame` (~3.8 FPS) for end-to-end demo video rendering[cite: 38].

---

## Repository Structure

```text
the-aerial-guardian/
├── visdrone_person_detection.ipynb      # End-to-end tiling, training (YOLOv8s-P2 + NWD), & eval
├── visdrone_person_evaluate_only.ipynb  # Evaluation & demo video generation using saved best.pt
├── index.html                           # GitHub Pages project website
├── Model output                         # Demo video (demo_out.mp4) and report figures
├── results/                             # metrics.json, metrics.csv, demo_detections.json, configs
└── report/                                # Project report (.docx/.pdf) and presentation (.pptx)
