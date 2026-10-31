# Publish VISIONFORGE without training or running Python on your computer

The ZIP already contains the finished model weights, test results, Grad-CAM exports, figures, screenshots, source code and PDF report. You **do not need to reproduce training** to show the project in your portfolio.

## 1. Upload to GitHub

1. Download and **extract** `VISIONFORGE_GitHub_Ready.zip` using your file manager.
2. Create a new **public** GitHub repository, for example `VISIONFORGE-Explainable-AI` (name is up to you).
3. On the new repository page, choose **Add file → Upload files**. Drag the *contents* of the extracted project folder so `README.md`, `visionforge/`, `web/`, `experiments/`, etc. appear at the repository root.
4. Commit all files to the `main` branch. Large checkpoint files are ~22 MB each; avoid accidentally excluding them. If browser upload is unreliable, use GitHub Desktop or a standard Git client. No project execution is necessary.
5. The README shows the actual screenshots and results immediately on GitHub.

GitHub does **not** automatically unpack a ZIP uploaded to a repository. The directory structure must appear as actual files in the repository.

## 2. Publish the interactive web dashboard

1. Open the GitHub repository's **Settings → Pages**.
2. In **Build and deployment**, choose **GitHub Actions** (not a branch or `/docs`).
3. Go to the **Actions** tab and, if prompted, enable workflows. The supplied `Publish research dashboard (GitHub Pages)` workflow will deploy the `web/` folder when the main branch is pushed or the workflow is manually run.
4. Return to **Settings → Pages** to find the public link after a successful deployment. The site operates with the precomputed experiment JSON and images, no backend.

### What is interactive?

The website provides genuine model comparison plots, model-specific confusion matrices, an interactive 12-example test gallery, view toggles between original image and CNN/ResNet Grad-CAM, probabilities and outcome details. It does **not** accept custom image uploads because static GitHub Pages has no PyTorch server.

## 3. Optional live uploaded-image inference

If you want website visitors to upload their own digit picture, connect this repo to [Streamlit Community Cloud](https://share.streamlit.io/) and choose `app.py` as the application file. Streamlit executes actual model predictions using the bundled checkpoints. It requires cloud Python dependencies but no local execution or retraining. Live uploads are only meaningful for handwritten digits consistent with the model's tiny training domain.

## 4. What not to claim

- The verified result is a small scikit-learn **digits** experiment, not a colour-image or high-resolution benchmark.
- ResNet-18 used **self-supervised 4-way rotation pretraining on digits**. It did **not** use pretrained ImageNet weights in the verified pilot.
- The optional `scripts/train_cifar10.py` is unexecuted. Don't present it as a finished CIFAR-10 experiment.
- Grad-CAM heatmaps are real gradients, but not proofs of why a model decided something.

For a full methods and outcomes review, read `docs/VISIONFORGE_Research_Report.pdf` and `experiments/offline_digits/EXPERIMENT.md`.
