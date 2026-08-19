# Partial-Point-Supervised Segmentation for Remote Sensing (LoveDA)

**Author:** Meer Muhammad

Training a semantic segmentation model using only **sparse point annotations**
instead of full pixel masks, via a custom **Partial Cross-Entropy (pCE) loss**,
evaluated on the real-world **LoveDA** land-cover dataset.

---

## Problem

Standard segmentation models need a complete pixel-level mask for every
training image. In many real annotation workflows, labelers instead mark
only a handful of individual **points** per image ("this pixel is forest,"
"that pixel is water") — far cheaper to collect, but a much weaker training
signal. This project asks: **can a segmentation network still be trained
effectively from sparse point labels alone?**

## What this project does

1. **Implements Partial Cross-Entropy (pCE) loss** — a loss function that
   computes cross-entropy only at annotated (point-labeled) pixels, giving
   exactly zero gradient elsewhere:

   ```
   pCE = Σ( CE(pred, GT) × MASK_labeled ) / Σ(MASK_labeled)
   ```

2. **Simulates point supervision on a real dataset (LoveDA)** by sampling a
   configurable number of points from each image's full ground-truth mask,
   using either:
   - **random** sampling, or
   - **class-balanced** sampling (guaranteed coverage of every class present)

3. **Trains a segmentation network** — U-Net with an ImageNet-pretrained
   ResNet-34 encoder — using only the sparse point labels and pCE loss.

4. **Runs controlled experiments** comparing random vs. class-balanced point
   sampling at matched annotation budgets, and reports the results.

---

## Repository contents

| File | Description |
|---|---|
| `Partial_Point_Segmentation.ipynb` | Self-contained proof-of-concept notebook (synthetic land-cover data, executed end-to-end offline) including the pCE loss with correctness/gradient checks, and the production PyTorch training code |
| `loveda_training.ipynb` (Kaggle) | Real-data pipeline: dataset loading from LoveDA, point simulation, U-Net + pCE training, and experiments |
| `Technical_Report.docx` | Report for the synthetic-data proof-of-concept experiments |
| `LoveDA_Technical_Report.docx` | Report for the real LoveDA experiments (random vs. class-balanced sampling) |
| `README.md` | This file |

---

## Dataset

**[LoveDA](https://github.com/Junjue-Wang/LoveDA)** (Land-cover Domain
Adaptive semantic segmentation) — 1024×1024 RGB satellite images across
Urban and Rural scenes, with 7 land-cover classes plus a no-data/ignore
class.

| Split | Images |
|---|---|
| Train | 2,522 |
| Val | 1,669 |
| Test (unlabeled, held out by LoveDA) | 1,796 |

**Class distribution** (sampled from training masks) — the dataset is
meaningfully imbalanced, which motivates the sampling-strategy experiment:

| Class | % of pixels |
|---|---|
| background | 34.74% |
| agriculture | 18.14% |
| forest | 14.80% |
| building | 10.64% |
| water | 6.36% |
| barren | 5.98% |
| road | 4.63% |
| no-data (ignored) | 4.72% |

---

## Method summary

- **Point simulation:** given a full mask, either sample `N` random pixels,
  or `K` points from every class present in the image ("class-balanced").
- **Loss:** `PartialCELoss` — a drop-in `nn.Module` replacement for
  `nn.CrossEntropyLoss`, masked to only labeled pixels.
- **Model:** `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
  classes=NUM_CLASSES)` — transfer learning via an ImageNet-pretrained
  encoder.
- **Training:** Adam, lr=1e-4, images/masks resized to 512×512 (masks use
  **nearest-neighbor** interpolation to preserve integer class labels —
  bilinear resizing corrupts labels at class boundaries).
- **Evaluation:** mean IoU over the 7 real classes (no-data excluded),
  computed on full validation masks.

---

## Running this yourself (Kaggle)

1. Create a new Kaggle Notebook, enable **GPU** and **Internet** in Settings.
2. Add the LoveDA dataset as an input (`+ Add Input` → search "LoveDA").
3. Install extra packages:
   ```bash
   pip install segmentation-models-pytorch albumentations -q
   ```
4. Collect file paths from `/kaggle/input/.../Train` and `/kaggle/input/.../Val`.
5. Build the point-supervised `SegmentationDataset`, `PartialCELoss`, and
   the U-Net model (see notebook for full code).
6. Train and evaluate with `evaluate_miou(..., verbose=True)` for per-class
   IoU breakdowns.
7. Save outputs to `/kaggle/working/` and commit the notebook to persist
   results.

---

## Key findings

**Experiment: random vs. class-balanced point sampling**, tested at two
matched annotation budgets (~10 and ~24 labeled pixels/image):

| Budget | Class-balanced mIoU | Random mIoU |
|---|---|---|
| ~10 px/image | 0.358 | **0.427** |
| ~24 px/image | 0.392 | **0.428** |

Contrary to the initial hypothesis, **random sampling outperformed
class-balanced sampling** on overall mIoU at both budgets — driven mainly by
stronger performance on dominant classes (background, agriculture, water),
since class-balanced sampling's fixed per-class cap under-supervises large,
easy classes. Class-balanced sampling's one consistent advantage was on
**barren** (rare and visually confusable with other surfaces), where it beat
random by 0.05–0.12 IoU at both budgets. **Road** showed no meaningful
difference between strategies at either budget, suggesting its difficulty is
more about thin/elongated object geometry than annotation strategy.

Full per-class breakdowns, charts, and discussion are in
`LoveDA_Technical_Report.docx`.

---

## Next steps

- Test an even smaller budget (e.g. 1 point/class) where rare-class omission
  under random sampling may become more severe.
- Try a hybrid strategy: a small guaranteed minimum per class plus
  additional points allocated proportional to class area.
- Run a point-density sweep (5 → full mask) on LoveDA to find where
  performance plateaus, as was done on synthetic data in the proof-of-concept
  notebook.
