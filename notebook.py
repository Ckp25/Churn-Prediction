# %%
# Downloading the dataset using kagglehub
# This script uses the kagglehub library to download the Telco Customer Churn dataset
#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("blastchar/telco-customer-churn")

#print("Path to dataset files:", path)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Remove warnings
import warnings
warnings.filterwarnings("ignore")

# %%
# Load the dataset as a pandas DataFrame
df = pd.read_csv(r'D:\Customer Churn\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head(2)

# %%
# Check the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)
# Check for null values
print("Null values in the DataFrame:\n", df.isnull().sum())

# %%
# Look at the data types of the columns
print("Data types of the columns:\n", df.dtypes)

# %%
# TotalCharges is a numerical column, but it is stored as an object type
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# %%
# Make two lists of categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
# Print the lists
print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# %%
# Check if SeniorCitizen is a true numeric column
df['SeniorCitizen'].value_counts()

# %%
# Remove SeniorCitizen column from numerical columns and add it to categorical columns
numerical_cols.remove('SeniorCitizen')
categorical_cols.append('SeniorCitizen')
# Print the updated lists
print("Updated Categorical columns:", categorical_cols)
print("Updated Numerical columns:", numerical_cols)

# %%
# Check for missing values in TotalCharges column
missing_total_charges = df['TotalCharges'].isnull().sum()
print("Missing values in TotalCharges column:", missing_total_charges)

# %%
# To fill in missing values in TotalCharges, check with MonthlyCharges and tenure
df[df['TotalCharges'].isna()][['tenure', 'MonthlyCharges']]


# %%
# Since tenure is 0 for all missing TotalCharges, we can fill these with 0
df['TotalCharges'].fillna(0, inplace=True)
# Verify that there are no more missing values in TotalCharges
print("Missing values in TotalCharges after filling:", df['TotalCharges'].isnull().sum())

# %% [markdown]
# Observe Numerical Columns First

# %%
# Obtain Histplots of numerical columns
for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# %%
# Obtain value counts of MonthlyCharges
monthly_charges_counts = df['MonthlyCharges'].value_counts().sort_index()
# Max and min values of MonthlyCharges
print("Max MonthlyCharges:", df['MonthlyCharges'].max())
print("Min MonthlyCharges:", df['MonthlyCharges'].min())

# %% [markdown]
# ### Observations: tenure
# 
# - Bimodal: spike near 0 and another around 70
# - Indicates new customers and long-term loyal ones
# - Hypothesis: new customers more likely to churn → to be verified in bivariate section
# - Logged full insight in `eda_log.md`
# 
# ### Observations: MonthlyCharges
# 
# - Bimodal: peaks around $20 and $80
# - Suggests two distinct user groups — minimal vs premium service usage
# - Hypothesis: churn might be higher in low-paying group → to be tested in bivariate
# - Detailed notes logged in `eda_log.md`
# 
# ### Observations: TotalCharges
# 
# - Originally stored as object → converted to float
# - 11 rows had NaNs (from `tenure == 0`)
# - These users had valid `MonthlyCharges`, but likely left before billing
# - Imputed TotalCharges = 0 and added `is_new_user` flag
# - Logged full logic in `eda_log.md`
# 
# 

# %% [markdown]
# Before moving on to categorical columns, need to decide encoding

# %%
# dataframe with only categorical columns
df_categorical = df[categorical_cols]
df_categorical.head()

# %%
# Checking if customerID is useful for analysis
# Split it into two parts: prefix and suffix
df['customerID_prefix'] = df['customerID'].str.split('-').str[0]
df['customerID_suffix'] = df['customerID'].str.split('-').str[1]
# Check the unique values in the prefix and suffix
print("Unique values in customerID prefix:", len(df['customerID_prefix'].unique()))
print("Unique values in customerID suffix:", len(df['customerID_suffix'].unique()))


# %%
# Drop the customerID, suffix, and prefix columns
df.drop(columns=['customerID', 'customerID_prefix', 'customerID_suffix'], inplace=True)

# %%
# Remove CustomerID from categorical columns
categorical_cols.remove('customerID')
df_categorical = df[categorical_cols]
df_categorical.head()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_categorical_countplots(df, categorical_cols, hue='Churn', cols_per_row=3, figsize_per_plot=(5, 4)):
    n = len(categorical_cols)
    rows = math.ceil(n / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(figsize_per_plot[0]*cols_per_row, figsize_per_plot[1]*rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'{col} Count Plot')
        axes[i].tick_params(axis='x', rotation=45)

    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show(block=True)

fig = plot_categorical_countplots(df, categorical_cols)
display(fig)


# %%
# Convert 'No internet service' to 'No' for internet-related columns
internet_related_cols = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

for col in internet_related_cols:
    df[col] = df[col].replace('No internet service', 'No')

# %%
fig = plot_categorical_countplots(df, categorical_cols)
display(fig)

# %%
# Obtain the percentage of unique values in each categorical column
def get_categorical_percentage(df, categorical_cols):
    percentages = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True) * 100
        percentages[col] = value_counts
    return percentages
percentages = get_categorical_percentage(df, categorical_cols)
percentages

# %% [markdown]
# Bivariate Analysis

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_categorical_countplots(df, categorical_cols, hue='Churn', cols_per_row=3, figsize_per_plot=(5, 4)):
    n = len(categorical_cols)
    rows = math.ceil(n / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(figsize_per_plot[0]*cols_per_row, figsize_per_plot[1]*rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, hue=hue, ax=axes[i])
        axes[i].set_title(f'{col} vs {hue}')
        axes[i].tick_params(axis='x', rotation=45)

    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show(block=True)  # <- guarantees rendering
plot_categorical_countplots(df, categorical_cols, hue='Churn')


# %%
def churn_percentage_by_category(df, categorical_cols, target_col='Churn'):
    for col in categorical_cols:
        print(f"\n{col} vs {target_col} (% within each category):")
        cross = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        display(cross.round(2))
churn_percentage_by_category(df, categorical_cols, target_col='Churn')

# %%
# Numerical columns wrt Churn
def plot_numerical_churn(df, numerical_cols, target_col='Churn'):
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=target_col, y=col)
        plt.title(f'{col} vs {target_col}')
        plt.xlabel(target_col)
        plt.ylabel(col)
        plt.show()
plot_numerical_churn(df, numerical_cols, target_col='Churn')

# %% [markdown]
# ### MonthlyCharges vs Churn
# 
# - Churned users have a **higher median MonthlyCharges (~$80)** than retained users (~$65)
# - Distribution is tighter among churners, concentrated in the $60–$95 range
# - Retained users include both high and low MonthlyCharges, suggesting:
#   - Low-charge customers likely retain due to low friction
#   - Mid-high charge customers churn more — possibly due to perceived cost-value imbalance
# - Supports the feature engineering flag: `is_low_charge = MonthlyCharges < 30`
# 
# ### Tenure vs Churn
# 
# - Churned users have a **very low median tenure (~10 months)**, while retained users average ~38 months
# - A large portion of churners leave within the **first year**, confirming early-stage dropoff
# - Very few churners reach long tenures (50+ months), highlighting strong loyalty among long-time users
# - Outliers beyond 70 months exist but are rare
# - Confirms usefulness of feature: `is_new_customer = tenure < 6`
# 
# ### TotalCharges vs Churn
# 
# - Churned users have **much lower total charges**, with a compressed distribution centered under ~$1,000
# - Retained users span a wider range (up to ~$8,500), reflecting extended billing cycles
# - While a few high-paying churners exist, they are outliers
# - Confirms: **high spend alone doesn’t prevent churn**, but **low lifetime value is highly predictive of it**
# - Suggests that even **customers with sunk cost (moderate TotalCharges)** may still churn if tenure is short
# 
# ### Behavioral Insight
# 
# - Low tenure and TotalCharges are strong churn indicators — they reflect **limited onboarding success**
# - Higher MonthlyCharges, despite sunk cost, may still lead to churn if perceived value is low
# - Users with higher disposable income may be **more willing to churn**, even after paying significant charges
# 
# 

# %%
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")

# %%
# Make categorical_cols without 'Churn' column
categorical_cols_no_churn = [col for col in categorical_cols if col != 'Churn']

# %%
# Encoding, making it ready to model
from sklearn.preprocessing import StandardScaler

# Step 1: Split off target
df1 = df.drop(columns='Churn')       # Features only
target = df['Churn'].copy()          # Preserve target separately

# Step 2: Scale numerical features
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df1[numerical_cols]),
    columns=numerical_cols
)

# Step 3: One-hot encode categorical features
df_encoded = pd.get_dummies(df1[categorical_cols_no_churn], drop_first=True).astype(int)

# Step 4: Combine all processed features with unchanged target
df_final = pd.concat([df_scaled, df_encoded, target], axis=1)

# Step 5: Save
df_final.to_csv("preprocessed_churn_data.csv", index=False)


# %%
df2 = pd.read_csv("preprocessed_churn_data.csv")
df2.head(2)

# %%
from sklearn.model_selection import train_test_split

df = pd.read_csv("preprocessed_churn_data.csv")
X = df.drop(columns='Churn')
y = df['Churn'].map({'No': 0, 'Yes': 1})  # Binary target

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

print("CV ROC-AUC scores:", scores)
print("Mean CV AUC:", scores.mean())


# %%
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# %%
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()


# %%
import matplotlib.pyplot as plt
import numpy as np

coef = model.coef_[0]
features = X.columns
top_idx = np.argsort(np.abs(coef))[::-1][:10]

plt.barh(features[top_idx], coef[top_idx])
plt.xlabel("Coefficient")
plt.title("Top Logistic Regression Feature Coefficients")
plt.gca().invert_yaxis()
plt.show()


# %%
# Plotting Precision-Recall Curve

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
# Plotting the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()

# %%
from sklearn.metrics import precision_recall_curve, f1_score

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Compute F1 scores
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid div/0
best_idx = f1_scores.argmax()
best_thresh = thresholds[best_idx]

print(f"Best F1 score: {f1_scores[best_idx]:.4f} at threshold: {best_thresh:.4f}")


# %%
y_pred_custom = (y_prob >= best_thresh).astype(int)


# %%
print(classification_report(y_test, y_pred_custom))

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best AUC:", grid.best_score_)
print("Best Params:", grid.best_params_)


# %%
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()


# %%
np.unique(y_train)


# %%
from sklearn.metrics import precision_score, recall_score, make_scorer

def weighted_fbeta(y_true, y_pred, beta=2):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)




# %%
fbeta_scorer = make_scorer(weighted_fbeta, beta=4)


# %%
# Optimizing the model with fbeta scoring
grid_fbeta = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    scoring=fbeta_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_fbeta.fit(X_train, y_train)
print("Best F-beta score:", grid_fbeta.best_score_)
print("Best Params for F-beta:", grid_fbeta.best_params_)

# Evaluate the best model
best_model_fbeta = grid_fbeta.best_estimator_
y_pred_fbeta = best_model_fbeta.predict(X_test)
y_prob_fbeta = best_model_fbeta.predict_proba(X_test)[:, 1]
print("Test ROC-AUC with F-beta:", roc_auc_score(y_test, y_prob_fbeta))

# %% [markdown]
# ## 🎯 Custom Metric: Business-Weighted Fβ Score
# 
# In churn prediction, not all errors are equal.
# 
# - **False Negative (FN)** → A churner is missed, resulting in lost revenue
# - **False Positive (FP)** → A loyal customer is wrongly targeted, wasting retention resources
# 
# To reflect this, we use a **weighted Fβ score**, which allows us to prioritize **recall** (catching churners) more than precision:
# 
# ### Fβ Score Formula:
# 
# \[
# F_{\beta} = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}
# \]
# 
# Where:
# - `P` = Precision
# - `R` = Recall
# - `β` (beta) adjusts the weight:  
#   - β = 1 → Standard F1 (precision = recall)  
#   - β > 1 → Recall is more important  
#   - β < 1 → Precision is more important
# 
# ---
# 
# ### Business-Aware β Calculation
# 
# We simulate a realistic cost scenario:
# 
# - `C_fn` = Revenue lost per churned customer ≈ **$65**  
# - `C_fp` = Cost of outreach or offer ≈ **$5**  
# - Then:
# 
# \[
# \beta = \sqrt{\frac{C_{fn}}{C_{fp}}} \approx \sqrt{\frac{65}{5}} \approx 3.6
# \]
# 
# We rounded this to **β = 4**, which biases the metric toward catching churners while allowing for some false positives.
# 
# ---
# 
# ### Final Results (β = 4)
# 
# - **Best Fβ Score**: 0.78 (on validation folds)
# - **Best Logistic Params**: `C=1`, `penalty='l1'`, `class_weight='balanced'`
# - **Test ROC-AUC**: 0.84  
# - **Churn Recall**: ~0.78  
# - **Churn Precision**: ~0.50–0.54
# 
# This custom metric ensures the model aligns with retention strategy and cost realities, not just abstract accuracy.
# 

# %%



