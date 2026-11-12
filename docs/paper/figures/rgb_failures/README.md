# Where an RGB camera fails: qualitative figures

Test images of HyperCOD on which the RGB pipeline fails and the pipeline on ten EC readings succeeds, plus the reverse cases.
Both pipelines use the same two stages and the same training recipe; only the input differs.

- **RGB pipeline (strict RGB chain).** A YOLO26s detector trained on the dataset's RGB camera image (`rgb_A`). Its boxes go to a SAM2-UNet trained on a three-channel pseudo-RGB image: the band means of the hyperspectral cube over 400–500, 500–600 and 600–700 nm (`seg_sam2unet_rgb_s*`).
- **EC pipeline (ours).** The same detector and segmenter trained on ten simulated electrochromic-sensor readings, greedy-selected and whitened (`sel10g_clean_A`, `seg_sam2unet_ec10_s*`).

Both are scored at level (c), the full image with each pipeline's own detector boxes. The operating point is conf ≥ 0.02 with at most five boxes per image. An image without any box counts as an empty mask.

This is the reference EC chain, whose segmenter is the same SAM2-UNet as the RGB pipeline's. The paper's final pipeline is a separate result (S 0.88 with SAM2.1 + box); see `../../results_summary.md` §3c.

## Files

| file | content |
|---|---|
| `fig_rgb_failures_main.pdf` / `.png` | main figure, 6 examples (PDF with embedded fonts; PNG at 600 dpi, 7.2 in wide) |
| `fig_rgb_failures_all17_part1.pdf` / `.png` | supplementary: the 10 images where the RGB detector returns no box |
| `fig_rgb_failures_all17_part2.pdf` / `.png` | supplementary: the 7 images where it boxes background or the segmenter misses a boxed object |
| `fig_ec_failures_reverse.pdf` / `.png` | supplementary: the 3 images where the EC pipeline fails and the RGB pipeline succeeds |
| `fig_rgb_failures_main_panels/` | every panel of the main figure as a separate PNG at native crop resolution, for re-layout. `panels.json` holds the crop window, the boxes in crop coordinates with confidences, IoU, S and failure mode |
| `rgb_vs_ec_per_image.csv` | all 70 test images: what each detector did, seed-mean IoU / S of both pipelines, seed-0 IoU |
| `select_rgb_failures.py`, `infer_rgb_vs_ec.py`, `render_rgb_failures.py` | the selection, the inference and the figures (see *Regenerate*) |

Each figure row is one test image with six panels on the same square crop. The crop holds the object and every box of both detectors. The panels are:

1. the RGB camera image
2. the ground truth
3. the RGB pipeline's detection boxes (orange)
4. the RGB pipeline's mask (probability > 0.5)
5. the EC pipeline's detection boxes (blue)
6. the EC pipeline's mask

The row label gives the test image and the failure mode, judged on the detector's own boxes:

- **no detection:** the detector returns no box.
- **box on background:** no box holds at least 25 % of the object.
- **boxed, not segmented:** a box holds the object, but the segmentation IoU is below 0.2.

## Selection rule

1. All 70 test images were ranked by the IoU of each pipeline, averaged over the three segmenter seeds (0, 1, 2). The detectors are single runs (seed 42).
2. "RGB fails, EC succeeds" means RGB IoU ≤ 0.2 and EC IoU ≥ 0.5. **17 images qualify.** The reverse holds for **3 images**. With a threshold of ±0.3 on the IoU difference, it is 20 against 4.
3. The supplementary figures show all 17 and all 3. No image was left out.
4. The main figure shows 6 of the 17. They cover the three failure modes and six different scenes: grass, shrub, dark grass, leaf litter, broad leaves and autumn leaves. Within each failure mode, images where the object is hard to see in the RGB image were preferred, and images with a conspicuous object (e.g. #196, #235) were not.
5. The panels show the seed-0 masks, and the IoU printed on them is seed 0's. The CSV gives the 3-seed means.

Failure modes among the 17:

- 10 no detection
- 5 box on background
- 2 boxed, not segmented

In the reverse cases, the EC detector returns no box once and boxes background twice.

## Numbers for the text (whole test set, 70 images)

| | RGB pipeline | EC pipeline (ours) | EC − RGB [95 % CI] |
|---|---|---|---|
| detector AP50 (checkpoints 59–99) | 0.497 | 0.739 | +0.242 [+0.112, +0.356] |
| objects fully inside an ROI (checkpoints 59–99) | 0.586 | 0.809 | |
| images without any box | 17 / 70 | 6 / 70 | |
| false-positive boxes | 45 / 133 | 27 / 116 | |
| end-to-end S (3 seeds) | 0.751 ± 0.004 | 0.849 ± 0.004 | +0.098 [+0.044, +0.148] |
| end-to-end IoU | 0.463 ± 0.013 | 0.668 ± 0.008 | +0.204 [+0.107, +0.299] |
| end-to-end weighted F | 0.497 ± 0.013 | 0.714 ± 0.006 | +0.217 [+0.112, +0.319] |

The intervals come from a paired image bootstrap with 2000 resamples. Inside a correct (oracle) box the two inputs segment alike: pseudo-RGB 0.888 ± 0.008 S against ec10 0.899 ± 0.007 (`results/seg/compare.json`, level (a)). The RGB pipeline therefore loses on finding the object, not on the mask.

## Caption drafts

**Main figure.** *Camouflaged objects that an RGB camera misses.* Each row is a HyperCOD test image. From left to right:

- the RGB camera image
- the ground-truth mask
- the RGB pipeline: a detector trained on the RGB image, then a segmenter on a pseudo-RGB composite inside its boxes
- our pipeline: the same detector and segmenter trained on ten EC-sensor readings

Detection panels show the boxes passed to the segmenter (confidence ≥ 0.02, at most five per image). Segmentation panels show the full-image mask with its IoU against the ground truth.

The RGB detector returns no box (#32, #214, #470), boxes background (#53, #6), or boxes the object without the segmenter recovering it (#98). The EC pipeline finds and segments every one of these objects. Over the 70 test images the RGB pipeline fails (IoU ≤ 0.2) while ours succeeds (IoU ≥ 0.5) on 17 images, and the reverse happens on 3 (Supplementary Fig. X).

**Supplementary figure (all 17 + reverse).** *All test images on which one pipeline fails and the other succeeds.* Panels as in Fig. Y. Parts 1–2 show the 17 images where the RGB pipeline fails (mean IoU over three segmenter seeds ≤ 0.2) and the EC pipeline succeeds (≥ 0.5). In 10 of them the RGB detector returns no box, in 5 it boxes background only, and in 2 the segmenter misses a boxed object. Part 3 shows the 3 images with the reverse outcome. The masks shown are those of seed 0.

## Regenerate

From the repository root:

```bash
python docs/paper/figures/rgb_failures/select_rgb_failures.py        # -> rgb_vs_ec_per_image.csv; prints the 17 + 3
CUDA_VISIBLE_DEVICES=1 python docs/paper/figures/rgb_failures/infer_rgb_vs_ec.py --images 196 301 32 92 111 77 214 53 6 95 88 235 204 470 251 281 98 258 212 191
cd docs/paper/figures/rgb_failures
python render_rgb_failures.py --images 32 214 470 53 6 98 --out fig_rgb_failures_main --panels
python render_rgb_failures.py --images 196 32 111 77 214 88 235 204 470 251 --out fig_rgb_failures_all17_part1
python render_rgb_failures.py --images 301 92 53 6 281 95 98 --out fig_rgb_failures_all17_part2
python render_rgb_failures.py --images 258 212 191 --out fig_ec_failures_reverse --reverse
```

`infer_rgb_vs_ec.py` recomputes the full-image probability maps exactly as `main_seg_eval.py` does and stores them in `results/paper_figures/rgb_failures/maps/` (git-ignored, 0.3 MB). It reproduces the evaluation's stored IoU and S to 0.0 on all 20 images. It needs the checkpoints `weights/seg_sam2unet_{rgb,ec10}_s0` and the ROI exports `results/det/rois_{rgb_A,sel10g_clean_A}_test.json`. The RGB images are read from `<data_path>/test/RGB/<id>.jpg`.
