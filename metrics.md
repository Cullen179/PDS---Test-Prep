### Accuracy

- **Rationale:** measures overall correctness; best when classes are **balanced** and false positives/false negatives have similar cost.
    
- **Interpretation:** “X% of predictions are correct.” But it can look high even if the model ignores a rare class.
- **Follow-up actions after interpreting:**
    - If accuracy is high but the minority class is important, also check **precision/recall/F1** and the **confusion matrix**.
    - If accuracy is low, inspect error types (FP vs FN) and look for segments where errors concentrate.

### F1 (usually F1 for the positive/minority class, or macro-F1)

- **Rationale:** balances **precision** (how many predicted positives are correct) and **recall** (how many true positives you found). Good for **imbalanced** data when both FP and FN matter.Precision=TP+FPTP,
    
- **Interpretation:** higher F1 means better trade-off between missing positives and over-flagging positives. Macro-F1 treats each class equally (useful when all classes matter).
- **Follow-up actions after interpreting:**
    - If **F1 is low**, check whether it’s driven by **low recall** (missing positives) or **low precision** (too many false alarms).
    - Adjust the **decision threshold** (not just the model) to trade precision vs recall.
    - Consider class weighting, resampling, or features aimed at minority-class separation.

### ROC-AUC

- **Rationale:** evaluates how well the model **ranks** positives above negatives across *all thresholds* (threshold-independent). Useful when you care about ranking/score quality.
- **Interpretation:** probability that a random positive gets a higher score than a random negative.
    - 0.5 = random, 1.0 = perfect ranking
- **Follow-up actions after interpreting:**
    - If ROC-AUC is good but F1/accuracy is poor, your model ranks well but your **threshold is wrong** → choose a threshold based on business cost or maximize F1/Youden’s J.
    - If ROC-AUC is near 0.5, the model has little separability → revisit features/model choice.
    - For **highly imbalanced** problems, also check **PR-AUC** (often more informative than ROC-AUC).

**Rule of thumb for choosing**

- Balanced classes + equal costs → **Accuracy**
- Imbalanced + care about positive detection quality → **F1 (or Precision/Recall)**
- Need good ranking / will pick threshold later → **ROC-AUC** (plus PR-AUC if very imbalanced)

`
### Conclusion and understanding of models results

A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. 

Most of the ML models works best when the number of classes are in equal proportion since they are designed to maximize accuracy and reduce error. Thus, they do not take into account the class distribution / proportion or balance of classes. In our dataset, the percentage of customer accepting the bank loan offered in campaign (class 1) is 9.6% whereas 90.4% of customers didn't accept the loan offered (class 0).

The confusion matrix is another metric that is often used to measure the performance of a classification algorithm, which contains information about the actual and the predicted class.

Metrics that can be calculated from confusion matrix:
* **Precision**: When it predicts the positive result, how often is it correct? i.e. limit the number of false positives.
* **Recall**: When it is actually the positive result, how often does it predict correctly? i.e. limit the number of false negatives.
* **f1-score**: Harmonic mean of precision and recall.

The confusion matrix for class 1 (Accepted) would look like:

|                        | Predicted: 0 (Not Accepted) | Predicted: 1 (Accepted)|
|------------------------|-----------------------------|------------------------|
|**Actual: 0 (Not Accepted)**| True Negatives              | False Positives        |
|**Actual: 1 (Accepted)**    | False Negatives             | True Positives         |

* **Precision would tell us cases where actually the personal loan wasn't accepted by the customer but we predicted it as accepted.**
* **Recall would tell us cases where actually the personal was accepted by the customer but we predicted it as not accepted.**

In our case, it would be recall that would hold more importance then precision. So choosing recall and f1-score which is the harmonic mean of both precision and recall as evaluation metric, particularly for class 1.

Further, AUC-ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, higher the AUC, better the model is at distinguishing between people accepting the loan and people not accepting the loan offered by the bank [source](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5).
Thus based on our evaluation metric, the scores of the models we tried are as below:

| Models      | Recall Score for Class 1 (%) | f1-score for Class 1 (%) | ROC AUC (%) | Accuracy (%) |
|-------------|--------------------------|----------------------|-----|----|
| **1. Logistic Regression** | 52 | 62 | 75 | 93.9 |
| **2. Logistic Regression with Hyperparameter Tuning** | 54 | 65 | 76 | 94.3 |
| **3. Logistic Regression with Hyperparameter Tuning and Oversampling** | 89 | 60 | 89 | 88.5 |
| **4. k-Nearest Neighbor** | 34 | 43 | 66 | 91 |
| **5. k-Nearest Neighbor with Hyperparameter Tuning** | 26 | 36 | 62 | 90.9 |
| **6. k-Nearest Neighbor with Hyperparameter Tuning and Oversampling** | 56 | 49 | 74 | 88.4 |

It can be seen that **Model 3** gives a better measures overall against others. 
`

### RMSE (Root Mean Squared Error)

- **What it measures:** typical error size, but **punishes big mistakes**.
- **Interpretation (beginner-friendly):**
    - Same unit as the target (e.g., dollars, kWh).
    - “On average, predictions are off by about **RMSE units**, with extra penalty for large misses.”
- **When to use:** big errors are especially costly (e.g., underestimating demand causes stockouts).
- **Follow-up actions if RMSE is high:**
    - Check **outliers** / rare cases driving large squared errors.
    - Do **segment error analysis** (which group has huge misses).
    - Try **target transform** (e.g., log) if variance grows with size.
    - Use models/features that capture **nonlinearity**.

### MSE (Mean Squared Error)

- **What it measures:** like RMSE but in **squared units**.
- **Interpretation:** harder to explain because units are squared (“$²”), but mathematically convenient.
- **When to use:** mostly for optimization/technical comparison; for communication, RMSE is usually clearer.
- **Follow-up actions:** same as RMSE (they rank models similarly).

### MAE (Mean Absolute Error)

- **What it measures:** typical error size, **treats all errors linearly** (more robust to outliers).
- **Interpretation:**
    - Same unit as target.
    - “Predictions are off by about **MAE units** on average.”
- **When to use:** you want a stable metric not dominated by a few extreme misses (messy real-world data).
- **Follow-up actions if MAE is high:**
    - Improve overall signal (better features, less noise).
    - If RMSE is okay but MAE is high → many medium errors; focus on general fit.
    - If MAE is low but RMSE is high → a few large errors; focus on outliers/segments.

**Quick diagnostic using both:**

- If **RMSE ≫ MAE** (e.g., RMSE/MAE > ~1.5): a small number of big errors are hurting you → handle outliers/rare segments.
- If **RMSE ≈ MAE**: errors are more evenly spread → improve baseline signal/features.

---

## Secondary metrics (use for context, not usually for final selection)

### R² (Coefficient of Determination)

- **What it measures:** how much variance you explain vs predicting the mean.
- **Interpretation:**
    - **1.0** = perfect predictions
    - **0.0** = no better than predicting the average every time
    - **< 0** = worse than predicting the average
- **When it helps:** comparing models on the same dataset; understanding “signal strength”.
- **Common beginner pitfall:** high R² doesn’t guarantee small errors in units you care about.
- **Follow-up actions if R² is low:**
    - The problem may be noisy / missing key predictors.
    - Add strong predictors, interactions, or a more flexible model.
    - Re-check data quality and leakage.

### Generalization (train vs test)

- **R² (Train) = 0.876**, **R² (Test) = 0.824**
    - The drop (**0.052**) is small → the model generalizes reasonably well (not heavily overfitting).
    - **Test R² = 0.824** means the model explains about **82% of the variation** in test scores.

### Error size (in “score points”)

- **MAE = 0.919** → typical prediction is off by about **0.9 points**.
- **RMSE = 1.446** → errors are about **1.45 points** on average, with **extra penalty for larger misses**.
- Because **RMSE > MAE**, you likely have a few larger errors (outliers) raising RMSE.

### Plots

- **Predicted vs Actual:** most points lie close to the diagonal → predictions track actual scores well. A few points far from the line indicate cases the model struggles with (especially at the low-score end).
- **Residuals distribution:** centered near **0** (good, low bias) but with a **left tail** (some large negative residuals) → a few cases where the model under/over-predicts by a lot.

## Which metric to focus on for school test score prediction

- **Primary: MAE** (most interpretable)
    - “On average, we’re off by ~0.9 marks.” This is easy to explain to teachers/stakeholders.
- **Also report: RMSE** if **large mistakes matter more**
    - Useful if being wrong by 4–5 marks is much worse than being wrong by 1 mark.
- **Secondary: R²** for context (how much variance you capture), but don’t rely on it alone because it’s not in score units.