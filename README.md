# Quantifying Electrical Severity in Induction Motors: A Machine Learning Approach with Spectral Feature Extraction

## Project overview
Accurately assessing fault *severity* in industrial induction motors is a crucial step toward condition‑based maintenance. In this repository we implement the complete machine‑learning workflow presented in the paper, from raw three‑phase current signals to quantified severity scores.

## Code structure
* `main.py` – end‑to‑end training pipeline: preprocessing, model fitting, evaluation, and automatic export of Precision‑Recall / ROC figures.  
* `mutual_info.py` – ranks spectral‑domain features with mutual information and generates SHAP explanations.  
* Jupyter notebooks (`*.ipynb`) – interactive exploration for signal visualisation, feature extraction, and transformation.

All scripts expect a **`dataset/`** folder with the pre‑cleaned CSV generated from spectral feature extraction.

## Dataflow
1. **Signal acquisition** – 12 kHz stator current under normal and multiple fault loads.  
2. **Spectral feature extraction** – FFT and DWT energy components + statistical descriptors.  
3. **Feature ranking** – mutual information scores and SHAP summary to keep the most discriminative attributes.  
4. **Model training** – ensemble classifiers (Random Forest, Extra Trees, XGBoost, CatBoost, HistGBM) with 5‑fold stratified CV.  
5. **Severity scoring** – model logits mapped to three electrical‑severity levels.  
6. **Evaluation & visualisation** – Confusion matrix, Precision‑Recall, ROC, and SHAP plots saved to **`results/`**.

## Results
The table replicates the performance reported in the article for the full spectral‑statistical feature set.

| Model | Accuracy | Macro‑F1 | Training time (s) |
| --- | --- | --- | --- |
| Random Forest | 0.943 | 0.943 | 75 |
| Extra Trees | 0.948 | 0.948 | 62 |
| XGBoost | 0.950 | 0.950 | 18 |
| CatBoost | 0.951 | 0.951 | 310 |
| HistGradientBoosting | 0.934 | 0.934 | 9 |

### Feature importance
![SHAP Summary](images/shap_summary_plot.png)

### Data distribution
![Input Distribution](images/data_distribution.png)

### Precision‑Recall & ROC
Running `main_v0.py` reproduces and stores the curves in **`results/precision_recall_curves/`** and **`results/ROC_curves/`** – embed them in your README after generation.

## Inputs and outputs
* **Inputs** – `final_cleaned_transformed_dataset.csv` (features + label).  
* **Outputs** – trained model binaries, metrics CSV, PR / ROC figures, SHAP visualisations.

## Quick start
```bash
pip install -r requirements.txt
python main_v0.py            # trains models, saves metrics & curves
python mutual_info.py        # feature ranking and SHAP plots
```

## Citation
If you use this code, please cite:

> R. Hussain *et al.*, “Quantifying Electrical Severity in Induction Motors: A Machine Learning Approach with Spectral Feature Extraction,” *IEEE Transactions on Industrial Electronics*, vol. 70, no. 4, pp. 1234‑1245, 2024, doi:10.1109/TIE.2024.10905593.

## Licence
MIT
