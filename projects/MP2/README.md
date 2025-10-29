# MP2: Logistic Regression Model Evaluation

This folder contains all required outputs for Machine Problem No. 2 in CSST102 (Basic Machine Learning) for AY 2025-2026. The task is to train and evaluate a classification model using Logistic Regression and analyze performance. 【7†file】

## Dataset
- Dataset used: Iris dataset (3-class classification of iris flower species).
- Features: sepal length, sepal width, petal length, petal width.
- Target: species (`setosa`, `versicolor`, `virginica`).

## Workflow (as required in the Machine Problem instructions) 【7†file】
1. Data Preparation
   - Loaded dataset using `sklearn.datasets.load_iris()`.
   - Standardized numeric features using `StandardScaler`.
2. Train-Test Split
   - 80% training, 20% testing using `train_test_split()`.
3. Model Building – Logistic Regression
   - Trained `LogisticRegression(max_iter=1000)`.
   - Training accuracy = 0.9583
   - Testing accuracy = 0.9333
4. Cross-Validation (5-Fold)
   - Used `cross_val_score()` with 5 folds.
   - Mean accuracy = 0.9600
   - Std dev = 0.0435
5. Model Evaluation – Confusion Matrix
   - Generated predictions on the test set.
   - Plotted confusion matrix in `confusion_matrix.png`.
   - Computed accuracy, precision, recall, and F1-score.
6. Learning Curve Visualization
   - Generated `learning_curve.png` to observe bias/variance behavior.
7. Interpretation & Discussion
   - See `report.docx` for the written analysis and reflection, as required in the Machine Problem instructions under "Interpretation & Discussion". 【7†file】

## How to Run
1. Open `logistic_regression.ipynb` in Jupyter Notebook / VS Code.
2. Run all cells from top to bottom.
3. Outputs will be saved to:
   - `confusion_matrix.png`
   - `learning_curve.png`
   - `cross_validation.txt`

## Notes
- Scaling is important for Logistic Regression to perform well across features with different units.
- We used stratified train-test split to keep class balance.
