# Mobile Price Classification

## Problem Statement

The objective of this project is to build a multi-class classification system to predict the price range of mobile phones based on their specifications. This is a supervised learning problem where mobile devices are categorized into four price ranges:

- **0**: Low Cost
- **1**: Medium Cost
- **2**: High Cost
- **3**: Very High Cost

The project involves implementing multiple classification algorithms, comparing their performance using various evaluation metrics, and deploying an interactive web application for model demonstration.

## Dataset Description

**Dataset**: Mobile Price Classification Dataset  
**Source**: [Kaggle - Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

### Dataset Statistics

- **Total Samples**: 2000
- **Features**: 20
- **Target Classes**: 4 (price_range: 0, 1, 2, 3)
- **Missing Values**: None

### Target Variable Distribution

All four price ranges are equally balanced with 500 samples each:

- **Low Cost (0)**: 500 samples (25%)
- **Medium Cost (1)**: 500 samples (25%)
- **High Cost (2)**: 500 samples (25%)
- **Very High Cost (3)**: 500 samples (25%)

## Models Comparision Table

![Models Evaluation](./images/Models%20Evaluation.jpeg)


# Model Performance Observations

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8300 | 0.9460 | 0.8349 | 0.8300 | 0.8317 | 0.7739 |
| KNN | 0.5000 | 0.7698 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest (Ensemble) | 0.8350 | 0.9508 | 0.8302 | 0.8350 | 0.8290 | 0.7817 |
| XGBoost (Ensemble) | 0.9225 | 0.9937 | 0.9226 | 0.9225 | 0.9225 | 0.8967 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved the best performance with 96.50% accuracy and highest MCC of 0.9534. Despite being a linear model, it demonstrates excellent classification capability, indicating strong linear separability between price ranges. The near-perfect AUC of 0.9987 shows superior ranking ability. Recommended for deployment due to high accuracy, fast inference, and model interpretability. |
| Decision Tree | Delivered moderate performance with 83.00% accuracy and MCC of 0.7739. The model is interpretable and captures non-linear patterns but shows signs of overfitting compared to ensemble methods. Good AUC of 0.9460 indicates decent probability estimates, though it struggles with precise class boundary decisions. Useful for feature importance analysis but not optimal for production. |
| KNN | Exhibited poor performance with only 50.00% accuracy, equivalent to random guessing in a 4-class problem. The lowest MCC of 0.3350 confirms its inability to learn meaningful patterns. Despite feature scaling, the curse of dimensionality with 20 features severely limits distance-based classification. This model is not suitable for mobile price prediction tasks. |
| Naive Bayes | Achieved respectable 81.00% accuracy with strong AUC of 0.9506, showing good probability calibration. The assumption of feature independence may not perfectly hold for mobile specifications, limiting accuracy. However, its computational efficiency and probabilistic outputs make it suitable for quick baseline predictions and scenarios requiring fast inference times. |
| Random Forest (Ensemble) | Demonstrated solid performance with 83.50% accuracy and MCC of 0.7817, slightly better than single decision tree. The ensemble approach reduces overfitting through bagging multiple trees. Good AUC of 0.9508 provides reliable probability estimates. While it handles non-linear relationships well, it's outperformed by simpler logistic regression, suggesting limited benefit from ensemble complexity for this dataset. |
| XGBoost (Ensemble) | Secured second-best performance with 92.25% accuracy and strong MCC of 0.8967. The gradient boosting effectively captures complex feature interactions with excellent AUC of 0.9937. Shows robust and balanced performance across all metrics. While more computationally intensive than logistic regression, it serves as a powerful alternative when capturing potential non-linear patterns is critical. |