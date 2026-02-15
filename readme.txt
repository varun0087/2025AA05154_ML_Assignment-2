# Mobile Price Classification

**Machine Learning Assignment 2**  
M.Tech (AIML/DSE) - BITS Pilani

---

## Problem Statement

The objective of this project is to build a multi-class classification system to predict the price range of mobile phones based on their specifications. This is a supervised learning problem where mobile devices are categorized into four price ranges:
- **0**: Low Cost
- **1**: Medium Cost  
- **2**: High Cost
- **3**: Very High Cost

The project involves implementing multiple classification algorithms, comparing their performance using various evaluation metrics, and deploying an interactive web application for model demonstration.

---

## Dataset Description

**Dataset**: Mobile Price Classification Dataset  
**Source**: Kaggle - [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

### Dataset Statistics
- **Total Samples**: 2000
- **Features**: 20
- **Target Classes**: 4 (price_range: 0, 1, 2, 3)
- **Missing Values**: None

### Features

| Feature | Description | Type |
|---------|-------------|------|
| battery_power | Total energy a battery can store (mAh) | Continuous |
| blue | Has bluetooth or not | Binary (0/1) |
| clock_speed | Speed at which microprocessor executes instructions | Continuous |
| dual_sim | Has dual sim support or not | Binary (0/1) |
| fc | Front Camera megapixels | Continuous |
| four_g | Has 4G or not | Binary (0/1) |
| int_memory | Internal Memory (GB) | Continuous |
| m_dep | Mobile Depth (cm) | Continuous |
| mobile_wt | Weight of mobile phone | Continuous |
| n_cores | Number of processor cores | Continuous |
| pc | Primary Camera megapixels | Continuous |
| px_height | Pixel Resolution Height | Continuous |
| px_width | Pixel Resolution Width | Continuous |
| ram | Random Access Memory (MB) | Continuous |
| sc_h | Screen Height (cm) | Continuous |
| sc_w | Screen Width (cm) | Continuous |
| talk_time | Longest time battery will last during call | Continuous |
| three_g | Has 3G or not | Binary (0/1) |
| touch_screen | Has touch screen or not | Binary (0/1) |
| wifi | Has wifi or not | Binary (0/1) |

### Target Variable Distribution

All four price ranges are equally balanced with 500 samples each:
- Low Cost (0): 500 samples (25%)
- Medium Cost (1): 500 samples (25%)
- High Cost (2): 500 samples (25%)
- Very High Cost (3): 500 samples (25%)

---

## Models Implemented

Six classification algorithms were implemented and evaluated:

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9650 | 0.9943 | 0.9651 | 0.9650 | 0.9649 | 0.9533 |
| Decision Tree | 0.8300 | 0.9088 | 0.8324 | 0.8300 | 0.8302 | 0.7734 |
| KNN | 0.5050 | 0.7107 | 0.5105 | 0.5050 | 0.5027 | 0.3400 |
| Naive Bayes | 0.8100 | 0.9627 | 0.8117 | 0.8100 | 0.8101 | 0.7467 |
| Random Forest (Ensemble) | 0.8350 | 0.9527 | 0.8350 | 0.8350 | 0.8348 | 0.7800 |
| XGBoost (Ensemble) | 0.9225 | 0.9877 | 0.9228 | 0.9225 | 0.9224 | 0.8967 |

---

## Model Performance Observations

### Model-wise Analysis

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved the highest performance across all metrics with 96.50% accuracy and MCC of 0.9533. Despite being a linear model, it performs exceptionally well on this dataset, suggesting strong linear separability between price ranges. The high AUC score (0.9943) indicates excellent class discrimination capability. This model is recommended for deployment due to its superior performance and computational efficiency. |
| Decision Tree | Delivered moderate performance with 83.00% accuracy and MCC of 0.7734. While interpretable and fast, it shows signs of overfitting compared to ensemble methods. The lower recall and precision values suggest it struggles with certain price range boundaries. Good for understanding feature importance but not optimal for production use. |
| KNN | Showed the poorest performance with only 50.50% accuracy and lowest MCC (0.3400). The scaled features did not help significantly, likely due to the curse of dimensionality with 20 features. Performance suggests that nearest neighbor relationships are not reliable predictors in this feature space. Not recommended for this classification task. |
| Naive Bayes | Achieved respectable 81.00% accuracy with excellent AUC (0.9627), indicating good probability calibration despite lower classification accuracy. The assumption of feature independence may not hold perfectly for mobile specifications, limiting its performance. However, its speed and probabilistic nature make it useful for baseline comparisons. |
| Random Forest (Ensemble) | Demonstrated solid performance with 83.50% accuracy and MCC of 0.7800. The ensemble approach helps reduce overfitting seen in single decision trees. Good balance between accuracy and interpretability through feature importance. However, it's outperformed by both logistic regression and XGBoost in this case. |
| XGBoost (Ensemble) | Second-best performer with 92.25% accuracy and strong MCC (0.8967). The gradient boosting approach effectively captures complex patterns in the data. Excellent AUC score (0.9877) demonstrates superior ranking ability. While computationally more expensive than logistic regression, it provides robust predictions and is a strong alternative for production deployment. |

### Key Insights

1. **Linear vs Non-linear**: Surprisingly, the linear Logistic Regression outperformed all complex ensemble methods, indicating that the price range classification problem has strong linear characteristics.

2. **Ensemble Performance**: XGBoost and Random Forest showed strong but not superior performance, suggesting that the additional complexity of ensemble methods may not be necessary for this particular dataset.

3. **Distance-based Learning**: KNN's poor performance indicates that simple distance-based approaches are inadequate for this multi-dimensional feature space.

4. **Probability Calibration**: Naive Bayes showed good AUC despite moderate accuracy, making it suitable for scenarios where probability estimates are more important than hard classifications.

5. **Recommendation**: For production deployment, **Logistic Regression** is recommended due to its superior accuracy, fast inference, and interpretability. **XGBoost** serves as a strong alternative when slightly lower accuracy is acceptable in exchange for capturing non-linear patterns.

---

## Project Structure

```
ml-assignment-2/
│
├── app.py                 # Streamlit web application
├── train_models.py        # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── models/               # Trained model files
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
└── data/                 # Data files
    ├── train.csv         # Training dataset
    ├── test.csv          # Test dataset
    └── model_comparison.csv  # Results comparison
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd ml-assignment-2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset and place `train.csv` in the project root directory

4. Train the models:
```bash
python train_models.py
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Usage Guide

### Using the Web Application

1. **Select Model**: Choose from 6 different classification models in the sidebar
2. **Upload Test Data**: Upload a CSV file with mobile specifications and price_range column
3. **View Results**: Analyze evaluation metrics, confusion matrix, and predictions
4. **Download Results**: Export predictions with confidence scores as CSV

### Test Data Format

Download test data from: [Google Drive Link](https://drive.google.com/file/d/1bfqCqGuSCkfpmTl-qGhoKPj1NDGUgZjq/view?usp=drive_link)

Your CSV must include these columns:
- 20 feature columns (battery_power, blue, clock_speed, etc.)
- 1 target column (price_range)

---

## Deployment

### Streamlit Community Cloud

This application is deployed on Streamlit Community Cloud:

**Live App**: [Your-Streamlit-App-URL]

To deploy your own version:

1. Push code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App"
5. Select your repository and branch
6. Choose `app.py` as the main file
7. Click "Deploy"

---

## Evaluation Metrics Explained

- **Accuracy**: Percentage of correctly classified instances
- **AUC (Area Under Curve)**: Measures the model's ability to distinguish between classes
- **Precision**: Ratio of true positives to total predicted positives
- **Recall**: Ratio of true positives to total actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure considering all confusion matrix elements

---

## Technologies Used

- **Python 3.8+**: Programming language
- **Scikit-learn**: Machine learning algorithms and metrics
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation
- **Joblib**: Model serialization

---

## Future Enhancements

- [ ] Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- [ ] Feature engineering and selection
- [ ] Cross-validation for more robust evaluation
- [ ] Deep learning models (Neural Networks)
- [ ] Real-time prediction API
- [ ] Model explainability using SHAP values

---

## Author

**Varun Sharma**  
M.Tech (AIML/DSE) - BITS Pilani  
Email: your.email@example.com

---

## Acknowledgments

- Dataset: Kaggle Mobile Price Classification Dataset
- BITS Pilani - Work Integrated Learning Programmes Division
- Course: Machine Learning (M.Tech AIML/DSE)