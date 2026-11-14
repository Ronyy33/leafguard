# LeafGuard: Problem Statement

## Business problem
Farmers and agronomy field workers often can't diagnose leaf diseases early. Expert visits are slow and costly, and a late diagnosis spreads infection and cuts yield. LeafGuard lets a non-technical user upload a photo of a leaf and get back, within a second:
- the most likely disease (or "healthy")
- a confidence score
- a heatmap showing which part of the leaf drove the decision

The system watches itself in production, and retrains when its inputs drift or its accuracy drops.

## Scope (v1)
- **Crops / classes (10):**
  - Tomato: healthy, Early blight, Late blight, Bacterial spot, Leaf mold
  - Potato: healthy, Early blight, Late blight
  - Apple: healthy, Apple scab
- **Input:** one JPG/PNG leaf photo, up to 5 MB.
- **Output:** class label, confidence, top-3 classes, and a Grad-CAM heatmap.
- **Deployment:** runs locally or on-premises with Docker Compose. No cloud services are used, per the course rules.

## Data
- **Source:** the PlantVillage dataset (Hughes & Salathé, 2015), color subset, public licence: https://github.com/spMohanty/PlantVillage-Dataset
- **Size:** 300 images per class, 3,000 images in total. The cap keeps CPU training tractable.
- **Format:** 256×256 RGB JPEG, one folder per class.
- **Known biases and limitations:**
  - Photos were taken in a lab with a uniform background, so real field photos (cluttered background, varied lighting, blur) are out of distribution. This is exactly why drift monitoring matters.
  - Each class has a single image source, so there is no geographic or seasonal diversity.
  - The class balance is artificial: we capped every class at 300.

## Success metrics
| Type | Metric | Target |
|---|---|---|
| ML | Macro-F1 on the held-out test set | ≥ 0.90 |
| ML | Per-class recall | ≥ 0.80 for every class |
| System | p95 latency of `/predict` | < 200 ms on CPU |
| System | API error rate (5xx) | < 1 % |
| Ops | Data pipeline throughput | Reported in images/sec |
| Ops | Detection of input drift (brightness/blur) | Alert within one rolling window of 50 requests |

These targets double as the **acceptance criteria** in the test plan.

## Lifecycle at a glance
1. Ray ingest
2. Validate
3. Preprocess, which also computes the baseline stats
4. Train (MLflow)
5. Evaluate
6. Register the model in the MLflow registry and promote it to `@production`
7. Serve through the FastAPI backend
8. Monitor with Prometheus and Grafana
9. When a drift or accuracy alert fires, Alertmanager triggers a retrain (`dvc repro`)

Every model is traceable to a Git commit, a DVC data hash and an MLflow run ID.
