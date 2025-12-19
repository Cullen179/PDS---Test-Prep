# Modelling Report — Classification and Regression (Student Performance)

## Dataset and Target

- Dataset: `student-data-25s3.csv`
- Primary target column: `G3` (final grade)
- Features: all other columns in the dataset.

## Problem Formulations

### 1) Classification (Binary)

**Goal:** predict whether a student “passes” based on final grade.

- Target definition (as used in the notebook):
  - $y=1$ if `G3 >= 10`, else $y=0$.

**Why this framing:**

- The pass/fail threshold is easy to interpret and is a common operational decision boundary.
- It converts a continuous outcome into a decision problem where metrics like precision/recall/ROC-AUC are meaningful.

### 2) Regression (Continuous)

**Goal:** predict the numeric final grade `G3`.

**Why this framing:**

- Preserves the full information in the outcome.
- Enables error-based evaluation (e.g., RMSE/MAE) and supports “how far off are we?” questions.

## Data Splitting and Validation

- Train/test split: 80/20 with `random_state=42`.
- For classification tuning: **StratifiedKFold** is used to preserve class balance in each fold.
- For regression tuning: **KFold** is used because the target is continuous.

## Preprocessing / Feature Engineering

A single preprocessing `Pipeline` is used so that transformations are learned on the training data only and applied consistently to validation/test data.

### Categorical features (nominal)

- **`SimpleImputer(strategy="most_frequent")`**
  **What it does:** fills missing category values with the most common category in the training data.
  **Rationale:** preserves a valid category and keeps the column usable without inventing new labels.
- **`OneHotEncoder(drop="first", handle_unknown="ignore")`**
  **What it does:** converts categories into 0/1 indicator columns.
  **Rationale:** makes nominal categories usable for models that require numeric inputs; `handle_unknown="ignore"` prevents errors for unseen test categories; `drop="first"` avoids redundant dummy columns (reduces multicollinearity in linear models).

### Ordinal features (ordered categories)

- **`OrdinalEncoder(...)`**
  **What it does:** maps ordered categories to ordered integers (e.g., low < medium < high).
  **Rationale:** keeps the **order information** that would be lost with one-hot encoding, enabling the model to learn monotonic patterns (higher level → higher/lower outcome).

### Numerical features

- **`SimpleImputer(strategy="median")`**
  **What it does:** fills missing numeric values with the median from training data.
  **Rationale:** median reflects a typical value and is less affected by extreme values than the mean.
- **`StandardScaler()`**
  **What it does:** standardizes each feature to mean 0 and standard deviation 1.
  **Rationale:** appropriate for many scale-sensitive models (e.g., linear/regularized models, SVM, PCA) because it puts features on a comparable scale.
- **`RobustScaler()`**
  **What it does:** scales using the median and IQR (interquartile range), reducing the influence of outliers.
  **Rationale:** preferred when numeric features are skewed or contain outliers, so scaling reflects the typical spread rather than extreme values.
- **`MinMaxScaler()`**
  **What it does:** rescales features into a fixed range (usually 0–1).
  **Rationale:** useful when you want bounded inputs and for some distance-based/gradient-based methods, but it is more sensitive to outliers than `RobustScaler`.

### Linear / Logistic Regression (incl. Ridge/Lasso)

- **Recommended scaler:** **StandardScaler**
- **Why:** these models are sensitive to feature magnitude (regularization and gradients). Standardizing makes coefficients comparable and training stable.
- **If strong outliers/skew:** use **RobustScaler** instead of StandardScaler.

### KNN / SVM

- **Recommended scaler:** **StandardScaler** (default best choice)
- **Why:** both rely on distances/margins; unscaled features can dominate and hurt performance.
- **If many outliers:** **RobustScaler** often performs better than StandardScaler.
- **If using RBF-SVM:** scaling is especially important; StandardScaler is the standard baseline.

### Tree-based models (Decision Tree, Random Forest, Gradient Boosting, XGBoost)

- **Recommended scaler:** **No scaling**
- **Why:** trees split by thresholds and are invariant to monotonic scaling; scaling rarely changes performance.
- **Focus instead:** handle missing values appropriately and control overfitting (depth, min_samples_leaf).

## Classification Modelling

### Models considered

1. **Logistic Regression (baseline)**
   - Strengths: fast, interpretable coefficients, good baseline for linear separability.
2. **Decision Tree (baseline)**
   - Strengths: captures non-linear interactions, minimal preprocessing sensitivity.
3. **K-Nearest Neighbours (baseline)**
   - Strengths: simple non-parametric baseline; benefits from scaling.
4. **Random Forest (baseline)**
   - Strengths: strong general-purpose classifier; handles non-linearities and interactions.

### Evaluation

Reported from the notebook evaluation helper:

- Accuracy on train and test.
- Confusion matrix.
- Classification report (precision/recall/F1).
- ROC curve and ROC-AUC when probability scores are available.

**Why these metrics:**

- Accuracy provides a simple overall summary.
- Confusion matrix and class-wise precision/recall/F1 highlight asymmetric errors (false positives vs false negatives).
- ROC-AUC assesses ranking quality across thresholds (useful when class balance is imperfect).

### Hyperparameter tuning and rationale

### Logistic Regression (tuned)

**Parameters explored (how chosen):**

- **`penalty`: `["l1", "l2"]`** — covers the two most common regularizers.
- **`C`: `[0.001, 0.01, 0.1, 1, 10, 100]`** — a **log-spaced range** to test very strong → very weak regularization.
- **`max_iter`: `[100, 200, 500]` (or higher)** — increased until the solver reliably converges.

**Cross-validation setup:**

- **`StratifiedKFold` (shuffled, fixed seed)** to keep class proportions consistent across folds.
- **Scoring: ROC-AUC** to compare models by **ranking quality** without fixing a threshold.

**Justification (why these):**

- `C` controls the bias–variance trade-off; using orders of magnitude avoids missing the right scale.
- `l1` can produce sparse coefficients (useful when many features may be irrelevant); `l2` is more stable under correlation.
- Stratification gives a fair estimate when the positive class is rarer.

---

### Decision Tree (tuned)

**Parameters explored (how chosen):**

- **`max_depth`:** e.g., `range(1, 30)` to span shallow (high bias) to deep (high variance) trees.
- **`min_samples_split`:** e.g., `range(2, 10)` to prevent overly specific splits.
- **`min_samples_leaf`:** e.g., `range(1, 9)` to avoid tiny leaves that memorize noise.
- **`max_features`:** e.g., `["sqrt", "log2", None]` or integer ranges to limit features considered per split.
- **`criterion`: `["gini", "entropy"]`** to test two common impurity measures.

**Cross-validation setup:**

- **Stratified CV** (same as above), scoring aligned to objective (e.g., ROC-AUC or F1 for imbalanced classes).

**Justification (why these):**

- Trees can overfit easily; depth and leaf constraints directly reduce variance and improve generalization.
- `max_features` adds randomness/regularization and can improve stability.
- Testing both criteria ensures the split rule isn’t a hidden bottleneck.

---

### KNN (tuned)

**Parameters explored (how chosen):**

- **`n_neighbors`:** odd values, e.g., `1..39` step 2, to explore local → smoother neighborhoods and reduce ties.
- **`weights`: `["uniform", "distance"]`** to compare equal voting vs stronger influence from closer neighbors.
- **`algorithm`: `["auto", "ball_tree", "kd_tree", "brute"]`** to select the most efficient neighbor search for the dataset.

**Cross-validation setup:**

- **Stratified CV**, scoring consistent with your evaluation metric.

**Justification (why these):**

- `k` controls the bias–variance trade-off: small `k` fits local noise; larger `k` smooths.
- Distance weighting can help when nearer points are more informative than farther ones.
- Algorithm choice mainly affects speed (and feasibility) depending on sample size and dimensionality.

### Feature importance / feature selection (classification)

- A tree-based selector (`SelectFromModel(RandomForestClassifier)`) is used to identify influential transformed features.

**Justification:**

- Random forests provide a quick importance signal even when the downstream classifier differs.
- This can reduce noise features and improve simpler models.

### Model selection decision

- The selected classification model should be the one with the best **cross-validated** performance (prefer ROC-AUC / F1 where appropriate) while also showing reasonable train/test consistency.
- If a tuned model performs worse than the baseline, it can indicate:
  - over-regularization / overly constrained hyperparameters,
  - noisy cross-validation estimates,
  - or the baseline model already being near an optimum for the dataset.

(Use the notebook outputs: `best_params_`, `best_estimator_`, plus test-set metrics to justify the final pick.)

## Regression Modelling

### Models considered

1. **Linear Regression (baseline)**
   - Strengths: simple, interpretable baseline.
2. **Ridge Regression**
   - Strengths: stabilizes coefficients under multicollinearity; improves generalization.
3. **Lasso Regression**
   - Strengths: can shrink some coefficients to zero (feature selection).

### Evaluation

Reported from the regression evaluation helper:

- $R^2$ (train/test)
- MSE / RMSE
- MAE
- Residual diagnostics:
  - predicted vs actual plot
  - residual distribution histogram

**Why these metrics:**

- RMSE penalizes larger errors more (useful when big misses are costly).
- MAE is more robust and easier to interpret as “average absolute error”.
- $R^2$ provides a normalized summary of explained variance.

### Hyperparameter tuning and rationale

#### Ridge / Lasso tuning

- Tune `alpha` over a log-scale range (e.g., several orders of magnitude).
- Cross-validation: `KFold` with shuffle and fixed seed.
- Scoring: negative RMSE (equivalently minimize RMSE).

**Justification:**

- Regularization strength is scale-sensitive and typically spans orders of magnitude.
- KFold is appropriate for continuous targets.

### Feature selection (regression)

Two approaches are demonstrated:

- `SelectFromModel(Lasso(alpha=...))`: selects features driven to non-zero by L1 regularization.
- `SelectFromModel(RandomForestRegressor, threshold="median")`: selects higher-importance transformed features.

**Justification:**

- Helps reduce overfitting and can simplify the model.
- Provides interpretability: identifies which transformed inputs matter most.

### Model selection decision

- Prefer models that:
  - minimize test RMSE/MAE,
  - show stable train vs test performance,
  - and produce residuals without strong systematic patterns.

(Use the notebook outputs from baseline and tuned models to justify the final selection.)

## Summary of Key Choices (Justifications)

- **Pipeline-based preprocessing:** prevents leakage and ensures repeatability.
- **Most-frequent / median imputation:** robust defaults for categorical/numerical missingness.
- **One-hot encoding + scaling:** broad compatibility across models (linear, KNN, regularized regression).
- **StratifiedKFold for classification:** stable validation under class imbalance.
- **Regularization (Ridge/Lasso):** controls overfitting and handles multicollinearity.

## What to Cite From the Notebook

To fully “close the loop” in the write-up, reference the printed outputs from:

- each baseline model’s test metrics,
- each tuned model’s `best_params_` and test metrics,
- and the final model choice rationale (trade-offs + observed performance).
