# Image Analysis — BMI Prediction from Facial Images

Research project investigating BMI estimation from facial images using classical ML and deep learning approaches. Conducted under the SWAN Lab, IIT Kharagpur (Dept. of CSE).

---

## Repository Structure

```
Image-Analysis-BMI/
├── bmi_ml_pipeline/          # Classical ML baseline (VisualBMI dataset)
└── bmi_dl_pipeline/
    ├── VisualBMI/            # DL experiments on small VisualBMI dataset
    └── Face2BMI/             # DL pipeline on large Face2BMI dataset (modular)
```

---

## Pipelines

### 1. `bmi_ml_pipeline` — Classical ML Baseline

**Dataset:** [Visual BMI](https://www.kaggle.com/datasets/abidur14004/visual-bmi) (~small scale, includes weight/height/gender metadata)

**Features:**
- HOG (Histogram of Oriented Gradients)
- Color statistics
- Grayscale intensity statistics
- Tabular biometrics (weight, height, gender)

**Models trained:** Linear Regression, Ridge, Random Forest, Gradient Boosting, AdaBoost, XGBoost

**Evaluation:** MAE, MSE, RMSE, R², MAPE

---

### 2. `bmi_dl_pipeline/VisualBMI` — Deep Learning on VisualBMI

**Dataset:** Visual BMI (~3,962 images)

**Models:** EfficientNetV2-S, ConvNeXt-Small, Swin-Tiny, ViT-B/16 (via `timm`)

**Key techniques:** z-score BMI normalization, CosineAnnealingLR, gradient clipping

| Model | MAE | R² |
|---|---|---|
| EfficientNetV2-S | — | — |
| ConvNeXt-Small | — | — |
| Swin-Tiny | — | — |
| **ViT-B/16** | **4.73** | **0.46** |

> ViT-B/16 was the best-performing model on this dataset.

---

### 3. `bmi_dl_pipeline/Face2BMI` — Production-Scale Modular Pipeline

**Dataset:** [face2bmi-alignedimgs](https://www.kaggle.com/datasets/) (~59,850 aligned facial images, `final.csv`)

**Architectures:** EfficientNet-B3, ViT-B/16, ConvNeXt-Base

**Pipeline modules:**

| File | Role |
|---|---|
| `face_detection.py` | Face detection and alignment preprocessing |
| `dataset.py` | Dataset loader, augmentations |
| `models.py` | Model definitions and backbone loading |
| `train.py` | Training loop, mixed precision, checkpointing |
| `evaluate.py` | Evaluation and metric computation |
| `model_selection.py` | Cross-architecture comparison |
| `utils.py` | Shared helpers, logging, reproducibility |

**Training setup:** Kaggle T4×2 GPU, mixed-precision training, models saved to a Kaggle output dataset for persistence.

**Design intent:** Backbones are structured for reuse as frozen encoders in a downstream multi-task model (BMI, puffiness, acne, stress prediction).

---

## Tech Stack

Python · PyTorch · timm · OpenCV · Scikit-learn · XGBoost · NumPy · Pandas · Matplotlib · Kaggle (T4×2)

---

## Future Work

- **Multi-task learning:** Unify BMI, puffiness, stress, and acne prediction under a shared encoder + task-specific heads
- **MediaPipe landmarks:** Incorporate facial geometry features (landmark distances, facial ratios) alongside image encodings to improve R²
- **Ensemble:** Combine ViT and ConvNeXt predictions via learned weighting for reduced MAE
- **Deployment:** Streamlit or FastAPI interface for real-time BMI estimation from uploaded images
- **Bias audit:** Evaluate model fairness across gender, age, and ethnicity subgroups
- **Cross-dataset generalization:** Test Face2BMI-trained models on VisualBMI and vice versa to assess distribution shift

---

## Author

**Kishlay** — B.Tech CSE, BIT Mesra | Research Intern, SWAN Lab, IIT Kharagpur
