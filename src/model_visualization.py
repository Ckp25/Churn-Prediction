import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models and test set
model_fbeta = joblib.load("models/logistic_model_fbeta.joblib")
model_f1 = joblib.load("models/logistic_model_f1.joblib")
df_test = pd.read_csv("data/test_data.csv")

X_test = df_test.drop(columns="Churn")
feature_names = X_test.columns

# Extract coefficients
coef_fbeta = model_fbeta.coef_[0]
coef_f1 = model_f1.coef_[0]

# Build comparison dataframe
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coef_Fbeta': coef_fbeta,
    'Coef_F1': coef_f1,
    'Abs_Fbeta': np.abs(coef_fbeta),
    'Abs_F1': np.abs(coef_f1),
    'Delta': coef_fbeta - coef_f1
})

# Sort by absolute delta
coef_df['Abs_Delta'] = np.abs(coef_df['Delta'])
coef_df_sorted = coef_df.sort_values(by='Abs_Delta', ascending=False)

print("Top differences in feature impact (F-beta vs F1):")
print(coef_df_sorted[['Feature', 'Coef_Fbeta', 'Coef_F1', 'Delta']].head(10))

# Save to CSV for audit
coef_df_sorted.to_csv("data/coef_comparison_fbeta_vs_f1.csv", index=False)
print(" Coefficient comparison saved to: data/coef_comparison_fbeta_vs_f1.csv")

# Plot top features that differ
top_diff = coef_df_sorted.head(10)
bar_width = 0.35
x = np.arange(len(top_diff))

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, top_diff['Coef_Fbeta'], bar_width, label='F-beta')
plt.bar(x + bar_width/2, top_diff['Coef_F1'], bar_width, label='F1')
plt.xticks(x, top_diff['Feature'], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.title("Top Feature Coefficient Differences: F-beta vs F1")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Highlight business interpretation
most_important_fbeta = coef_df.sort_values(by='Abs_Fbeta', ascending=False).head(5)
most_important_f1 = coef_df.sort_values(by='Abs_F1', ascending=False).head(5)

print("\n[Business Insight] Top Features (F-beta):")
print(most_important_fbeta[['Feature', 'Coef_Fbeta']])

print("\n[Business Insight] Top Features (F1):")
print(most_important_f1[['Feature', 'Coef_F1']])
