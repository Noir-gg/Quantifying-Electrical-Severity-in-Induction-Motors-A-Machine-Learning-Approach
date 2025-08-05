import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
shap.initjs()


file_path = "../dataset/final_cleaned_transformed_dataset.csv"
df = pd.read_csv(file_path)
X = df.drop("label", axis=1)
y = df['label']

mutual_info = mutual_info_classif(X, y, discrete_features="auto")
mutual_info_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mutual_info})
mutual_info_df = mutual_info_df.sort_values(by='Mutual Information', ascending=False)
mutual_info_df.to_csv("Mutual_info.csv", index=False)

model = xgb.XGBClassifier()
model.fit(X, y)

plt.figure(figsize=(20, 12))
xgb.plot_importance(model, importance_type='gain')
plt.savefig("xgb_feature_importance.png")
plt.close()  # Close the figure after saving to prevent display overlap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.plots.force(shap_values[0:100])


