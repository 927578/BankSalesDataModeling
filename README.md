# 🏦 Bank Marketing Campaign Classification

This project uses **decision tree** and **random forest** classifiers to predict whether a client will subscribe to a term deposit based on a range of features from a bank marketing dataset. The modeling workflow includes **data preprocessing**, **label encoding**, **hyperparameter tuning**, and **performance evaluation** using cross-validation.

## 📁 Dataset

- **File**: `bank_data_coded.csv`
- **Source**: UCI Bank Marketing Dataset
- **Delimiter**: `;`
- **Target Variable**: `y` (1 = subscribed, 0 = not subscribed)
- **Features Include**:
  - Demographics: `age`, `job`, `marital`, `education`
  - Financial info: `default`, `balance`, `housing`, `loan`
  - Contact info: `contact`, `day`, `month`, `duration`
  - Campaign stats: `campaign`, `pdays`, `previous`, `poutcome`

## 🧹 Data Preprocessing

- Encoded all **categorical variables** using `LabelEncoder`
- Saved the processed dataset as `bank_data_coded.csv`
- Split into features (`X`) and target (`y`)

## 🧠 Models Used

### 🌳 Decision Tree

- Explored `max_depth` from 1 to 20
- 5-fold cross-validation
- Tracked:
  - Training accuracy
  - Test accuracy
  - Precision
  - Recall
- Plotted accuracy vs depth
- Output:
  - Best precision and its depth
  - Best recall and its depth

### 🌲 Random Forest

- Performed grid search with cross-validation
- Parameter grid:
  - `n_estimators`: [10, 50, 100]
- Evaluation metric: accuracy
- Output:
  - Best number of trees
  - Accuracy across folds

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- Cross-validation scores (mean over folds)

## 📦 Libraries Used

```python
pandas
numpy
matplotlib
sklearn (LabelEncoder, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV, cross_validate, metrics)
