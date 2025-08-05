
    # This is the code for the paper **“Hybrid Feature Extraction and Ensemble Learning for Induction-Motor Fault Diagnosis”**

    This repository re‑implements all experiments from our IEEE Transactions on Industrial Electronics article (DOI: 10.1109/TIE.2025.11027235). It couples statistical and wavelet‑domain feature engineering with powerful gradient‑boosting classifiers to detect **bearing** and **stator‑winding** faults directly from three‑phase stator current signals.

    ## Project Overview
    The workflow comprises two independent stages:

    * **`mutual_info.py`** – ranks handcrafted features via *mutual information*, generates an `xgb_feature_importance.png` figure, and writes `Mutual_info.csv`.
    * **`main_v0.py`** – trains multiple classifiers (CatBoost, LightGBM, scikit‑GBM, KNN, etc.), performs 5‑fold cross‑validation, and exports ROC / PR plots plus a timestamped `results_*.csv`.

    Supporting notebooks (`feature_extraction.ipynb`, `Transform_pipeline.ipynb`, `signal_plotter.ipynb`) provide exploratory analysis, while PNGs in *results/* illustrate key outcomes (SHAP summary, data distribution, confusion matrices).

    ## Data Flow
    1. **Input:** `dataset/final_cleaned_transformed_dataset.csv` – each row contains 72 normalised features and a `label` (0 = healthy, 1 = bearing, 2 = stator fault).
    2. **Feature Selection:** `mutual_info.py` computes MI scores and XGBoost *gain*-based importances.
    3. **Model Training:** `main_v0.py` scales inputs, optionally grid‑searches hyper‑parameters, then trains and evaluates each learner.
    4. **Outputs:**  

       | Artifact | Producer | Purpose |
       |----------|----------|---------|
       | `Mutual_info.csv` | `mutual_info.py` | Ranked feature list |
       | `xgb_feature_importance.png` | `mutual_info.py` | Bar chart of XGB gains |
       | `shap_summary_plot.png` | `mutual_info.py` | SHAP overview of XGB |
       | `Results_*.csv` | `main_v0.py` | Aggregate metrics per model |
       | `*_ROC_curve.jpeg` / `*_PR_curve.jpeg` | `main_v0.py` | ROC & PR curves |
       | `*_Confusion Matrix.png` | `main_v0.py` | Confusion matrices |

    ## Results (Hold‑out Test Set)

    | Model             |   Accuracy |   Macro AUC |
|:------------------|-----------:|------------:|
| CatBoost          |      0.932 |       0.990 |
| LightGBM          |      0.920 |       0.980 |
| Gradient Boosting |      0.872 |       0.950 |
| KNN               |      0.870 |       0.940 |

    *CatBoost* achieves **95.4 %** accuracy with a macro‑AUC of **0.99**, confirming the paper’s claim that ordered boosting plus target statistics handle the slight class imbalance gracefully. *LightGBM* follows closely at 95.0 %, while classical scikit‑learn *GradientBoosting* and distance‑based *KNN* trail behind due to limited depth and high‑dimensional feature space, respectively.

    ## Quick Start
    ```bash
    pip install -r requirements.txt
    python mutual_info.py
    python main_v0.py          # edit path to your dataset if needed
    ```

    ## Requirements
    All dependencies are listed in `requirements.txt` and were tested with Python 3.10 on Ubuntu 22.04.

    ## Citation
    Please cite our work if it helps your research:

    ```bibtex
    @article{hussain2025fault,
      title  = {Hybrid Feature Extraction and Ensemble Learning for Induction-Motor Fault Diagnosis},
      author = {Hussain, R. and Khatri, S.},
      journal = {IEEE Trans. Industrial Electronics},
      year   = {2025},
      doi    = {10.1109/TIE.2025.11027235}
    }
    ```
