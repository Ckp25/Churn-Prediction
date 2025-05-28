## Feature: MonthlyCharges

- Distribution: Bimodal
  - Peak 1: ~$20 (likely minimal service)
  - Peak 2: ~$80 (likely bundle or premium plans)
- Hypothesis: Low-tier users may churn due to dissatisfaction or lack of perceived value.
- TODO: Cross MonthlyCharges with Churn using:
  - `sns.histplot(hue='Churn')` or boxplot
  - Also group by Contract type (see if short-term low spenders churn more)
- Potential Feature Engineering:
  - Binary flag: `is_low_charge = MonthlyCharges < 30`
  - Segment: low, mid, high charge buckets


  ## Feature: Tenure

- Distribution: Bimodal with:
  - Peak at 0–5 months (new customers)
  - Rise at 60–72 months (long-term customers)
- Hypothesis: 
  - New users churn early due to unmet expectations
  - Long-term users are retained
  - Mid-range users (12–48 months) may be undecided or influenced by new offers
- TODO:
  - Analyze churn % across tenure bands
  - Cross `tenure` with `Churn`, `Contract`, and `PaymentMethod`
- Feature Engineering Ideas:
  - Bin into tenure categories (e.g., `new`, `mid`, `loyal`)
  - Create `is_new_customer = tenure < 6`


## Feature: TotalCharges

- Initial dtype: object → due to blank strings in rows where `tenure == 0`
- Fixed by: `pd.to_numeric(errors='coerce')` → resulted in 11 NaNs
- Verified those rows:
  - All had `tenure == 0` and valid (non-zero) `MonthlyCharges`
  - Implies: customers signed up but never completed a billing cycle
- Action:
  - Imputed `TotalCharges = 0` for those users
  - Added `is_new_user = (tenure == 0)` as a potential churn flag
- Hypothesis:
  - These users may churn disproportionately (early quitters)
  - Flag could be predictive of high onboarding churn


## Feature: customerID

- Format: xxxx-yyyy (split by hyphen)
- Extracted:
  - `cust_prefix`: numeric part before hyphen
  - `cust_suffix`: alphanumeric part after hyphen
- Findings:
  - `cust_prefix` had 5084 unique values
  - `cust_suffix` had 7040 unique values
  - Both show very high cardinality (~1:1 with dataset size)
  - No meaningful grouping or correlation with churn found
- Action:
  - Dropped `customerID` entirely from the dataset
  - Rationale: acts as a unique identifier, not a predictive feature


## Feature: OnlineSecurity, OnlineBackup, StreamingTV, StreamingMovies, TechSupport, DeviceProtection

- Original state:
  - Contained 'No internet service' as a separate category
  - These features are only applicable to users with InternetService ≠ 'No'
- Action:
  - Replaced 'No internet service' → 'No' across all six columns
  - Rationale: from a modeling and interpretability standpoint, no internet = no service
  - Simplifies all features into binary: 'Yes' / 'No'
  - Reduces categorical clutter and avoids false multicollinearity in modeling


## Feature: MultipleLines

- Contains 'No phone service' as third category
- This means user opted for internet-only plan
- Decision:
  - Kept 'No phone service' as a distinct category
  - Did NOT map to 'No' — preserves meaningful differentiation from single line users


### Category Proportion Summary

- `SeniorCitizen`, `PhoneService`: strongly imbalanced (83–90% in one class)
- `InternetService`, `Contract`, `PaymentMethod`: show meaningful variation
- `OnlineSecurity`, `TechSupport`, `DeviceProtection`: majority 'No', potentially important churn predictors
- `Churn`: 26.5% → confirms dataset imbalance (to be handled in modeling)
- Will keep all features for now, but some (e.g., `PhoneService`) may have low model importance



## eda_log: Categorical Features vs Churn

### Process
- Identified 16 categorical features (binary, nominal, or ordinal)
- Used `pd.crosstab(..., normalize='index')` to compute churn percentages within each category
- Visualized class-level churn using `sns.countplot` for qualitative support

---

### Findings (Summarized)

#### Contract
- Month-to-month: 42.7% churn  
- One year: 11.3%, Two year: 2.8%  
**→ Strongest categorical signal — short-term plans strongly linked to churn**

#### PaymentMethod
- Electronic check: 45.3% churn  
- Others (auto-pay, mailed): 15–19%  
**→ Manual payment is a churn proxy**

#### InternetService
- Fiber optic: 41.9%  
- DSL: 18.9%, No internet: 7.4%  
**→ Fiber users churn more — likely high-cost plans**

#### TechSupport and OnlineSecurity
- 'No' → ~31% churn  
- 'Yes' → ~15%  
**→ Lack of support/security strongly correlates with churn**

#### PaperlessBilling
- Yes: 33.6%, No: 16.3%  
**→ Digital-only users churn more**

#### Dependents and Partner
- No dependents → 31% churn, Yes → 15%  
- No partner → 33%, Yes → 19%  
**→ Family responsibilities may correlate with retention**

#### StreamingTV / StreamingMovies
- 'Yes' → ~30% churn, 'No' → ~24%  
**→ Optional media services may reflect user segment differences (e.g., younger or price-sensitive users)**

#### SeniorCitizen
- Senior (1): 41.7%, Non-senior (0): 23.6%  
**→ Higher churn risk among senior citizens**

#### Gender, PhoneService, MultipleLines
- Minimal churn differences across classes  
**→ Low signal strength — may be dropped or deprioritized during modeling**

---

### Interpretation Summary
- Several service-related and demographic features show strong class-level churn separation
- These features are correlational, not causal, but highly valuable for prediction
- Will retain all categorical features for modeling, with appropriate encoding
- Consider dropping or merging low-signal variables like `Gender`, `PhoneService` if necessary


### MonthlyCharges vs Churn

- Churned users have a **higher median MonthlyCharges (~$80)** than retained users (~$65)
- Distribution is tighter among churners, concentrated in the $60–$95 range
- Retained users include both high and low MonthlyCharges, suggesting:
  - Low-charge customers likely retain due to low friction
  - Mid-high charge customers churn more — possibly due to perceived cost-value imbalance
- Supports the feature engineering flag: `is_low_charge = MonthlyCharges < 30`


### Tenure vs Churn

- Churned users have a **very low median tenure (~10 months)**, while retained users average ~38 months
- A large portion of churners leave within the **first year**, confirming early-stage dropoff
- Very few churners reach long tenures (50+ months), highlighting strong loyalty among long-time users
- Outliers beyond 70 months exist but are rare
- Confirms usefulness of feature: `is_new_customer = tenure < 6`

### TotalCharges vs Churn

- Churned users have **much lower total charges**, with a compressed distribution centered under ~$1,000
- Retained users span a wider range (up to ~$8,500), reflecting extended billing cycles
- While a few high-paying churners exist, they are outliers
- Confirms: **high spend alone doesn’t prevent churn**, but **low lifetime value is highly predictive of it**
- Suggests that even **customers with sunk cost (moderate TotalCharges)** may still churn if tenure is short

### Behavioral Insight

- Low tenure and TotalCharges are strong churn indicators — they reflect **limited onboarding success**
- Higher MonthlyCharges, despite sunk cost, may still lead to churn if perceived value is low
- Users with higher disposable income may be **more willing to churn**, even after paying significant charges


### Custom Metric Evaluation: Fβ Score

- Standard F1 treats precision and recall equally
- However, in churn prediction, missing a churner (false negative) is **much costlier** than incorrectly targeting a loyal customer (false positive)
- We introduced the **Fβ score**, a weighted harmonic mean of precision and recall, to reflect this cost imbalance

#### Assumed Business Cost Estimates:

- Revenue lost per churned customer (C_fn): **$65**
- Cost per outreach or incentive (C_fp): **$5**
- Derived importance weight:

\[
\beta = \sqrt{\frac{C_{fn}}{C_{fp}}} \approx 3.6
\]

#### Model Optimization

- Optimized logistic regression using `GridSearchCV` with Fβ (β = 4) as scoring
- Resulting model used `L1` regularization for sparsity
- Fβ validation score: **0.78**
- ROC-AUC (test): **0.84**

This reflects a business-aware evaluation framework, where recall is correctly emphasized for retention planning.

