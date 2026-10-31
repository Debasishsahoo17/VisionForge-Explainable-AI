# Model cards

## Model 1 — CustomCNN (saved)

- **Task:** 10-class classification of small monochrome handwritten digits.
- **Training:** AdamW; 14 supervised epochs; validation-selected checkpoint, 11th epoch.
- **Provenance:** Trained here from random initialization on scikit-learn digits training split only.
- **Checkpoint:** `experiments/offline_digits/artifacts/custom_cnn.pt`.
- **Outcome:** 311/315 test images correct (98.730% accuracy); see `results.json` for precision, recall, F1, ROC-AUC, ECE, confidence interval and per-class breakdown.
- **Usage:** only handwritten digit images in the training domain; not suitable for medical, biometric, safety-critical or commercial decisions.

## Model 2 — Rotation-pretrained ResNet-18 (saved)

- **Task:** Same 10-class digit recognition task.
- **Source learning:** Four-way rotations of train-set images only; 2 epochs. **This is not ImageNet transfer**.
- **Downstream:** Replace 4-way source head with a 10-class head; train all layers for 6 epochs; keep best on validation split.
- **Checkpoints:** `rotation_source_weights.pt` (source) and `rotation_resnet18.pt` (downstream).
- **Outcome:** 309/315 test images correct (98.095% accuracy); see JSON for additional metrics.
- **Usage:** interpret in light of small-data domain and unequal pretraining budget.

## Shared risks and caveats

- 8×8 source images are digitally upsampled to 32×32 for network input; there is no additional spatial resolution.
- Softmax confidence need not represent actual probability of correctness, particularly for uploads unlike the benchmark.
- Grad-CAM is a gradient-based spatial attribution heuristic, not causal evidence, instance segmentation, or a proof of correct reasoning.
- No demographic inference or facial recognition is performed.
- Reproducibility is CPU-testable, but small numerical differences may occur across hardware/PyTorch versions.
- Pretraining phase uses 0, 90, 180, 270° transforms. Rotation of digits such as 6 and 9 can change interpretation; the source task measures orientation, not digit identity.
- The source pretraining objective uses no held-out labels, but the same underlying training images; this is a valid within-domain transfer pilot rather than cross-dataset transfer.

## Model file integrity

All predictions were generated from the stored, validation-selected checkpoints and saved to compressed `.npz` arrays with test indices. No accuracy is hardcoded in training or evaluation functions. The static dashboard reads `web/assets/results.json` and precomputed test-case images.
