# Reproducibility and audit guide

The repo ships with all inference weights for both **executed offline models**, test prediction probabilities, split indices, metrics, Grad-CAM images, actual-browser UI screenshots and one-command experiment scripts.

## Evidence audit

1. Read `experiments/offline_digits/results.json` for metrics and metadata.
2. Read `experiments/offline_digits/EXPERIMENT.md` for scope and limitations.
3. Open `experiments/offline_digits/artifacts/*_predictions.npz` for `y_true`, `probabilities`, and `test_indices`.
4. Model checkpoints are `*.pt` dictionaries containing the full weights and architecture metadata; use PyTorch `torch.load(..., weights_only=True)`.
5. Browse `web/` for static precomputed test-inference explorer; no model training or active backend is required to view it.
6. The 6 screenshots in `docs/screenshots/` were captured from the actual HTML, CSS and JavaScript running in Chromium with the same asset JSON injected offline. They are not illustrative mockups or AI-generated images.

## Reproduce the offline benchmark

```bash
python -m pip install -r requirements.txt pytest
python scripts/run_experiment.py --cnn-epochs 14 --pretext-epochs 2 --transfer-epochs 6
python scripts/export_assets.py
pytest -q tests
```

This re-trains from random seeds on CPU and re-exports evidence. GPU execution may change minor numerical values. The recorded run used PyTorch CPU and the provided split seed.

## Run an optional *actual* ImageNet transfer experiment

```bash
python scripts/train_cifar10.py --train-limit 12000 --cnn-epochs 10 --transfer-epochs 5
```

This script **requires an internet download** of CIFAR-10 and the torchvision ResNet18 `IMAGENET1K_V1` weights (about 45 MB). The script uses a frozen pretrained image backbone with a trained 10-class head, plus a random-initialized custom CNN. The train split, validation selection, test probabilities and reported metrics follow the same conventions. **It has not been executed and has no included claims, plots or metrics.** When executed, its separate results are written to `experiments/cifar10_imagenet/`, not mixed with offline digit results.

## Publishing to GitHub Pages without running locally

The file `.github/workflows/pages.yml` publishes `web/` as a static website on every push to the `main` branch. In your repository settings, open **Pages → Build and deployment → GitHub Actions**. Once the first workflow succeeds, GitHub displays a public site link. The static site features 12 precomputed test examples and real graphs; it intentionally does **not** offer file uploads. For live upload inference, deploy `app.py` via Streamlit Community Cloud from the same GitHub repo.

## Artifact semantics

- Confidence scores come from softmax of checkpoint logits.
- ROC-AUC uses multiclass one-vs-rest macro averaging (well-defined because each test class is present).
- ECE uses 10 bins of prediction confidence.
- CIs are percentile bootstraps of held-out test accuracy (400 resamples).
- Grad-CAM is produced with actual backward hooks for a chosen target class and is *not* an object localization ground truth.
