# Dataset card — Optical Recognition of Handwritten Digits (scikit-learn)

**Source:** [scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html); derived from the Optical Recognition of Handwritten Digits dataset.

**Samples:** 1,797 labelled images. **Classes:** digits 0 through 9. **Input:** 8×8 single-channel images at source, pixels 0–16. **Network preprocessing:** float32, divided by 16, bilinear resize to 32×32, no augmentation in downstream training.

**Splits:** random seed 42; stratified at two stages: 1,168 training, 314 validation, 315 test. Test indices are saved alongside predictions. The exact same split is used to compare both networks.

**Data accessibility:** loaded directly from scikit-learn, with no download needed. Raw images are not vendored in this GitHub repository. Selected, transformed test images are included only to support qualitative result inspection.

**Limitations:** small resolution, clean backgrounds and a narrow category set. It is an excellent controllable test for a research workflow, not evidence of state-of-the-art vision performance. The benchmark does not represent real-world colour photographs or a deployment population.
