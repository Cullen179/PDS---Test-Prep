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
    