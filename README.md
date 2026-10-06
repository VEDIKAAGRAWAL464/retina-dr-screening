# 👁️ Diabetic Retinopathy Screening Assistant

**SDG 3: Good Health and Well-being** | IBM SkillsBuild Machine Learning & Applied AI Internship 2026 (AICTE x BharatCares)

🔗 **Live demo:** https://retina-dr-screening-9vvxfqwgafafh55et7bayn.streamlit.app/

> ⚠️ Screening-support prototype only. Not a medical diagnosis tool.

## Problem
Diabetic retinopathy (DR) is a leading cause of preventable blindness, but eye specialists are scarce, especially in rural India. Early screening can prevent vision loss.

## Solution
A deep learning system that reads a retina (fundus) photo and:
1. Grades DR severity (0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative)
2. Shows a Grad-CAM heatmap of the regions that drove the prediction
3. Gives a triage message: **No referable DR / Referable DR / Uncertain, consult a doctor** (confidence threshold 0.6)

## Dataset
Kaggle APTOS 2019 Blindness Detection (3,662 labeled retina images), stratified 80/10/10 train/val/test split.

## Pipeline
- Preprocessing: black-border crop, square padding, resize to 256x256, Gaussian-blur contrast enhancement
- Augmentation: flips, rotation, brightness/contrast jitter
- Imbalance handling: class-weighted cross-entropy
- Models: baseline 4-block CNN vs EfficientNet-B0 (transfer learning, mixed precision, cosine LR)
- Explainability: Grad-CAM
- Deployment: Streamlit Community Cloud

## Results (held-out test set, 367 images)
| Model | Accuracy | QWK |
|---|---|---|
| Baseline CNN | 0.706 | 0.810 |
| **EfficientNet-B0** | **0.766** | **0.891** |

**Referable DR (grade 2+) detection:** Sensitivity 0.933, Specificity 0.954, ROC-AUC 0.989.
With the 0.6 confidence threshold, 84% of cases are auto-decided at 0.828 accuracy and 0.941 sensitivity; the rest are referred to a doctor.

## Limitations
- Strong at healthy vs referable, but neighbouring grades get confused (Moderate vs Severe). Severe-class F1 is only 0.31 due to limited samples.
- Trained on a single dataset; no external validation yet.
- Not clinically validated and not a substitute for an eye doctor.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Sample images are in `/samples`.

## Tech stack
Python, PyTorch, timm, OpenCV, scikit-learn, Streamlit

**Author:** Vedika Agrawal
