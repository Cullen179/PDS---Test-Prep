# Data Science Test Preparation - Template Code Blocks

## Table of Contents

1. [Import Libraries](#1-import-libraries)
2. [Data Loading](#2-data-loading)
3. [Data Exploration (EDA)](#3-data-exploration-eda)
4. [Data Quality Checks](#4-data-quality-checks)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Feature Engineering](#6-feature-engineering)
7. [Data Visualization](#7-data-visualization)
8. [Train-Test Split](#8-train-test-split)
9. [Classification Models](#9-classification-models)
10. [Model Evaluation](#10-model-evaluation)
11. [Hyperparameter Tuning](#11-hyperparameter-tuning)
12. [Handling Imbalanced Data](#12-handling-imbalanced-data)

---

## 1. Import Libraries

```python
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Styling
import matplotlib.style as style
style.use('fivethirtyeight')

# Statistics
from scipy.stats import zscore, norm

# Sklearn - Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

# Sklearn - Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

# Sklearn - Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Sklearn - Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

# Imbalanced Learning
from imblearn.over_sampling import SMOTE

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Random seed for reproducibility
np.random.seed(42)
```

---

## 2. Data Loading

```python
# Load CSV file
df = pd.read_csv('filename.csv')

# Load from UCI Repository
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=503)  # Replace with dataset ID
X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)

# Display first rows
df.head()
```

---

## 3. Data Exploration (EDA)

### Basic Info

```python
# Shape
print(f"Shape: {df.shape}")

# Column info
df.info()

# Statistical summary (numerical)
df.describe()

# Statistical summary (categorical)
df.describe(include='object')

# Column names
print(df.columns.tolist())

# Data types
print(df.dtypes)
```

### Unique Values Summary

```python
# Quick unique values check
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Detailed summary
summary_df = pd.DataFrame({
    "column": df.columns,
    "dtype": [df[c].dtype for c in df.columns],
    "nunique": [df[c].nunique() for c in df.columns],
    "null_count": [df[c].isnull().sum() for c in df.columns],
    "sample_values": [df[c].unique()[:5] for c in df.columns]
})
summary_df
```

### Value Counts

```python
# Single column
df['column_name'].value_counts()

# With percentages
df['column_name'].value_counts(normalize=True) * 100

# Including NaN
df['column_name'].value_counts(dropna=False)
```

---

## 4. Data Quality Checks

### Check Null Values

```python
# Count nulls per column
print(df.isnull().sum())

# Percentage of nulls
print((df.isnull().sum() / len(df) * 100).round(2))

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.title('Missing Values Heatmap')
plt.show()
```

### Check Duplicates

```python
# Count duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# View duplicate rows
df[df.duplicated()]

# Remove duplicates
df = df.drop_duplicates()
```

### Check Domain Violations

```python
def domain_violation_count(df, col, condition):
    """Check for domain constraint violations"""
    mask = ~df[col].apply(condition)
    count = mask.sum()
    print(f"{col}: {count} values violate domain constraint")
    return df.loc[mask, col]

# Example usage
domain_violation_count(df, 'Age', lambda x: 0 <= x <= 120)
domain_violation_count(df, 'Experience', lambda x: x >= 0)
```

### Check Outliers (IQR Method)

```python
def detect_outliers_iqr(df, col):
    """Detect outliers using IQR method"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers (range: {lower:.2f} - {upper:.2f})")
    return outliers

# Apply to numerical columns
for col in df.select_dtypes(include=[np.number]).columns:
    detect_outliers_iqr(df, col)
```

### Check Outliers (Z-Score Method)

```python
def detect_outliers_zscore(df, col, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(zscore(df[col].dropna()))
    outliers = df[col].dropna()[z_scores > threshold]
    print(f"{col}: {len(outliers)} outliers (z-score > {threshold})")
    return outliers
```

---

## 5. Data Preprocessing

### Handle Missing Values

```python
# Drop rows with null
df = df.dropna()

# Fill with mean/median/mode
df['col'] = df['col'].fillna(df['col'].mean())
df['col'] = df['col'].fillna(df['col'].median())
df['col'] = df['col'].fillna(df['col'].mode()[0])

# Forward/backward fill
df['col'] = df['col'].ffill()
df['col'] = df['col'].bfill()
```

### Handle Outliers

```python
# Cap outliers using IQR
def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
    return df
```

### Encoding Categorical Variables

```python
# Label Encoding (for ordinal)
le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['col'], drop_first=True)

# Manual mapping
mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df['col'] = df['col'].map(mapping)
```

### Scaling/Normalization

```python
# Standardization (Z-score)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numerical_cols])

# Min-Max Normalization
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df[numerical_cols])
```

---

## 6. Feature Engineering

### Create New Features

```python
# Binning
df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])

# Interaction features
df['income_per_family'] = df['Income'] / df['Family']

# Log transformation (for skewed data)
df['log_income'] = np.log1p(df['Income'])
```

### Feature Selection

```python
# Correlation with target
correlations = df.corr()['target'].abs().sort_values(ascending=False)
print(correlations)

# Drop low correlation features
threshold = 0.1
low_corr_features = correlations[correlations < threshold].index.tolist()
df = df.drop(columns=low_corr_features)
```

---

## 7. Data Visualization

### Distribution Plots

```python
# Histogram for all numerical columns
df.hist(bins=30, figsize=(15, 10), layout=(4, 4))
plt.tight_layout()
plt.show()

# Single distribution with KDE
plt.figure(figsize=(8, 5))
sns.histplot(df['col'], kde=True)
plt.title('Distribution of Column')
plt.show()
```

### Count Plots (Categorical)

```python
# Single column
plt.figure(figsize=(8, 5))
sns.countplot(x='col', data=df)
plt.title('Count Plot')
plt.xticks(rotation=45)
plt.show()

# Multiple count plots
n_cols = 4
cols = df.select_dtypes(include='object').columns
n_rows = math.ceil(len(cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
for ax, col in zip(axes.ravel(), cols):
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=45)
for ax in axes.ravel()[len(cols):]:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

### Box Plots (Outliers)

```python
# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_cols])
plt.xticks(rotation=45)
plt.title('Box Plot - Outlier Detection')
plt.show()

# Box plot by category
plt.figure(figsize=(8, 5))
sns.boxplot(x='category', y='numerical', data=df)
plt.show()
```

### Correlation Heatmap

```python
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

### Scatter Plot

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x='col1', y='col2', hue='target', data=df)
plt.title('Scatter Plot')
plt.show()
```

### Pair Plot

```python
sns.pairplot(df, hue='target', diag_kind='kde')
plt.show()
```

---

## 8. Train-Test Split

```python
# Define features and target
X = df.drop('target', axis=1)
y = df['target']

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Target distribution:\n{y_train.value_counts(normalize=True)}")
```

---

## 9. Classification Models

### Logistic Regression

```python
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]
```

### K-Nearest Neighbors

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
```

### Naive Bayes

```python
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
```

### Decision Tree

```python
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

### Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

---

## 10. Model Evaluation

### Classification Report

```python
print(classification_report(y_test, y_pred))
```

### Confusion Matrix

```python
# Basic confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

### Individual Metrics

```python
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

### ROC Curve

```python
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Cross-Validation

```python
# K-Fold Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

---

## 11. Hyperparameter Tuning

### Grid Search CV

```python
# Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)
print(f"Best params: {grid_lr.best_params_}")
print(f"Best score: {grid_lr.best_score_:.4f}")

# KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)
```

### Finding Optimal K for KNN

```python
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, marker='o')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('KNN - Accuracy vs K')
plt.show()

print(f"Best K: {k_range[np.argmax(scores)]}, Accuracy: {max(scores):.4f}")
```

---

## 12. Handling Imbalanced Data

### Check Class Balance

```python
print(y.value_counts())
print(y.value_counts(normalize=True))
```

### SMOTE (Synthetic Minority Oversampling)

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original: {y_train.value_counts().to_dict()}")
print(f"Resampled: {pd.Series(y_train_resampled).value_counts().to_dict()}")
```

### Class Weight (for Logistic Regression)

```python
lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
```

---

## Quick Reference - Metrics Interpretation

| Metric        | Use When                                                  |
| ------------- | --------------------------------------------------------- |
| **Accuracy**  | Balanced classes                                          |
| **Precision** | False positives are costly (e.g., spam detection)         |
| **Recall**    | False negatives are costly (e.g., disease detection)      |
| **F1-Score**  | Imbalanced classes, need balance between precision/recall |
| **ROC-AUC**   | Compare models, threshold-independent evaluation          |

---

## Quick Reference - Variable Types

| Type         | Description             | Examples                |
| ------------ | ----------------------- | ----------------------- |
| **Nominal**  | Categories, no order    | Gender, Color, ID       |
| **Ordinal**  | Categories with order   | Education level, Rating |
| **Interval** | Numerical, no true zero | Temperature (°C)        |
| **Ratio**    | Numerical, true zero    | Age, Income, Height     |

---

## Common Data Quality Checklist

- [ ] Check shape and dimensions
- [ ] Check data types
- [ ] Check null/missing values
- [ ] Check duplicates
- [ ] Check unique values
- [ ] Check domain violations
- [ ] Check outliers
- [ ] Check class balance (for classification)
