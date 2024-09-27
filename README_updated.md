
# GST Hackathon Project - Predictive Modelling and Data Analysis

## Project Overview
This project was developed for the GST Hackathon and focuses on data analysis and predictive modeling using a dataset containing 900,000 individual data points with 21 attributes. These attributes include demographic data, transactional data, categorical, and continuous variables.

The project is aimed at building robust classification models using machine learning techniques to predict target variables and optimize performance based on various metrics.

## Features
- **Dataset**: 900,000 records, 21 features per record.
- **Tech Stack**:
  - Data processing libraries: Numpy, Pandas, Matplotlib
  - Machine Learning libraries: Scikit-learn (DecisionTreeClassifier, GradientBoostingClassifier, XGBClassifier), RandomForest, KMeans
  - Scaling: RobustScaler for handling outliers
  - Metrics: accuracy_score, roc_auc_score, log_loss, f1_score, precision, recall, confusion matrix visualization, etc.
  - Advanced imputation using MICE Algorithm for handling missing data
  - Feature Selection and Clustering: K-Means and Mutual Information used to identify important features and reduce data dimensionality.
  
## Data Preprocessing and Cleaning
1. **Data Cleaning**:
   - Dropped irrelevant columns
   - Handled missing values using the MICE algorithm
   - Outlier detection using Z-score analysis
2. **Data Visualization**:
   - Box plots for feature distribution analysis
   - Scatter plots for feature correlation
3. **Feature Selection**:
   - Mutual information-based ranking
   - Top features identified and used for training the final model

## Models and Training
### 1. **Decision Tree Classifier**:
   - Analyzed max_depth impact on accuracy.
   - Achieved high training and test accuracy with minimal overfitting.

### 2. **Gradient Boosting Classifier**:
   - Trained with varying n_estimators.
   - Achieved optimal complexity and high accuracy without overfitting.

### 3. **XGBoost Classifier**:
   - Trained with varying estimators, reached over 96% accuracy.
   - Confusion matrices were used for model evaluation.

### 4. **Stacking Model**:
   - Combined Decision Tree and Gradient Boosting Classifier using Logistic Regression as a meta-model.
   - Improved model accuracy by stacking multiple learners.

## Performance Metrics
Performance was evaluated using multiple metrics, including:
- **Accuracy**: Evaluated on both training and test sets.
- **Precision, Recall, F1-Score**: For class imbalance and model performance.
- **AUC-ROC**: Measures the model's ability to distinguish between classes.
- **Log Loss**: Penalizes incorrect confident predictions.
- **Balanced Accuracy**: Adjusts accuracy for imbalanced datasets.

### Decision Tree Classifier Performance
- Train Accuracy: 0.9772
- Test Accuracy: 0.9760
- F1 Score: 0.8797
- AUC-ROC: 0.9567

### Gradient Boosting Classifier Performance
- Train Accuracy: 0.9735
- Test Accuracy: 0.9734
- F1 Score: 0.8645
- AUC-ROC: 0.9423

### XGBoost Classifier Performance
- Train Accuracy: 0.9636
- Test Accuracy: 0.9634
- F1 Score: 0.7785
- AUC-ROC: 0.8390
