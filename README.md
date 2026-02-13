# End-to-End Machine Learning Pipeline – Breast Cancer Classification

## 1. Project Overview
This project builds an **end-to-end machine learning pipeline** to classify breast tumors as malignant or benign using the Breast Cancer Wisconsin dataset from `scikit-learn`.  
The goal is to simulate a production-style workflow with proper preprocessing, model training, evaluation, and saving the final pipeline as a reusable artifact.

---

## 2. Dataset

- Source: `sklearn.datasets.load_breast_cancer`.  
- Samples: 569  
- Features: 30 numeric features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. 
- Target:
  - 0 → malignant
  - 1 → benign[web:26][web:29]

The dataset is already clean and contains only numerical values, which makes it suitable for demonstrating an ML pipeline.

---

## 3. Tools and Libraries

- Python  
- NumPy, Pandas  
- Scikit-learn: `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`, model evaluation metrics.
- Joblib for saving the trained pipeline.

---

## 4. Approach and Workflow

The project follows a typical end-to-end ML workflow:

1. **Load data**  
   - Load the Breast Cancer dataset using `load_breast_cancer` and convert it to a Pandas DataFrame for easier exploration.
2. **Feature–target split**  
   - Separate features `X` and target `y` (`malignant` vs `benign`).

3. **Identify feature types**  
   - Since this dataset is fully numeric, all columns are treated as numerical features.  
   - The code is written generically to also handle categorical features if added later (via `ColumnTransformer`).

4. **Preprocessing with ColumnTransformer**  
   - Numerical pipeline: `StandardScaler` to normalize numerical features.  
   - Categorical pipeline: `OneHotEncoder(handle_unknown="ignore")` (kept for future extensibility).  
   - `ColumnTransformer` combines these into a single preprocessing step applied inside the pipeline.
5. **Build full ML pipeline**  
   - Create a `Pipeline` with two main steps:  
     - `preprocessor` → the `ColumnTransformer`  
     - `model` → `LogisticRegression(max_iter=1000)`.
   - This ensures the same preprocessing is applied during both training and inference and helps prevent data leakage.

6. **Train–test split**  
   - Split the data into training and test sets using `train_test_split` with stratification to preserve class balance.

7. **Model training**  
   - Fit the pipeline (`clf.fit(X_train, y_train)`), which first fits the preprocessors on the training data and then trains the logistic regression model.
8. **Evaluation**  
   - Generate predictions on the test set.  
   - Compute evaluation metrics: accuracy, precision, recall, F1-score and view the full classification report.

9. **Saving the pipeline**  
   - Save the entire trained pipeline as `breast_cancer_pipeline.pkl` using `joblib.dump`, so it can be loaded and used directly for predictions later.

---

## 5. Results

On the held-out test set, the model achieved:

- **Accuracy:** `0.9824`  
- **Precision:** `0.9861`  
- **Recall:** `0.9861`  
- **F1-score:** `0.9861`  

Classification report:

```text
               precision    recall  f1-score   support

           0       0.98      0.98      0.98        42
           1       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
These scores show that the pipeline performs very well on both malignant and benign classes.

6. Repository Structure
.
├── ML_pipeline.ipynb          # Main notebook with entire pipeline workflow
├── breast_cancer_pipeline.pkl # Saved trained pipeline (preprocessing + model)
├── AIML_task_15.pdf           # Task description / assignment PDF
└── README.md                  # Project documentation (this file)

7. How to Run This Project
  1. Clone the repository
  git clone https://github.com/PranithaBokketi/task15-end-to-end-machine-learning-pipeline.git
  cd task15-end-to-end-machine-learning-pipeline

  2. Create environment and install dependencies

  Install the required libraries (example using pip):

  pip install numpy pandas scikit-learn joblib

  4. Open the notebook
  jupyter notebook ML_pipeline.ipynb

Run all the cells to:

Load the dataset

Build the preprocessing and model pipeline

Train the model

Evaluate using accuracy, precision, recall, and F1-score

Save the trained pipeline as a .pkl file

8. Using the Saved Pipeline
After training, you can load the saved pipeline and make predictions on new data:

python
import joblib
import pandas as pd

# Load saved pipeline
pipeline = joblib.load("breast_cancer_pipeline.pkl")

# Example: new sample(s) with the same feature columns as the original dataset
# new_data should be a DataFrame with 30 feature columns
# new_data = pd.DataFrame([...], columns=feature_names)

predictions = pipeline.predict(new_data)
The pipeline automatically applies all preprocessing steps before making predictions, which is similar to how models are used in production systems.

9. Learning Outcomes
From this task, I practiced:

Building a complete ML pipeline combining preprocessing and model in scikit-learn.

Using ColumnTransformer and Pipeline to avoid data leakage and keep code modular.

Evaluating a classification model with multiple metrics (accuracy, precision, recall, F1-score).

