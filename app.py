import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import plotly.graph_objects as go
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

# Custom CSS for clean minimal UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        padding: 0.5rem 0 0.3rem 0;
        margin: 0;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.95;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .stSelectbox label {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
    }
    .download-section {
        background: #f8fafc;
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px dashed #cbd5e1;
        text-align: center;
        margin: 0.5rem 0 1rem 0;
    }
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Paths
MODEL_DIR = "model"
DATA_DIR = "data"

# Load scaler
@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

scaler = load_scaler()

# Title - More compact
st.markdown('<h1 class="main-title">Mobile Price Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ML Assignment 2 • M.Tech (AIML/DSE) • BITS Pilani</p>', unsafe_allow_html=True)

# Sidebar - Clean and minimal
with st.sidebar:
    st.markdown("### Model Selection")
    
    model_name = st.selectbox(
        "Choose a classifier:",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Dataset Information")
    st.markdown("""
    **Features:** 20  
    **Samples:** 2000  
    **Classes:** 4 price ranges
    
    - **0:** Low Cost
    - **1:** Medium Cost
    - **2:** High Cost
    - **3:** Very High Cost
    """)

# Model Files Mapping
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# Load selected model
@st.cache_resource
def load_model(model_file):
    return joblib.load(os.path.join(MODEL_DIR, model_file))

model = load_model(model_files[model_name])

# Main Content - Compact
st.markdown(f"### Current Model: **{model_name}**")

# Download Dataset Section - Compact
st.markdown('<div class="download-section">', unsafe_allow_html=True)
st.markdown("#### 📥 Download Test Dataset")

# Google Drive download link
drive_link = "https://drive.google.com/uc?export=download&id=1bfqCqGuSCkfpmTl-qGhoKPj1NDGUgZjq"

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.markdown(f'<a href="{drive_link}" target="_blank"><button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.6rem 1.5rem; border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 600; cursor: pointer; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);">Download Test CSV</button></a>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Upload test CSV - Right below download section
uploaded_file = st.file_uploader(
    "Upload your test CSV file (must include price_range column)",
    type=["csv"],
    help="Upload test data with 20 features and price_range target column"
)

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f"✓ Dataset loaded successfully • {len(df)} samples")
    
    # Show data preview
    with st.expander("View Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Prepare data
    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    
    # Scale only when required
    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_processed = scaler.transform(X)
    else:
        X_processed = X
    
    # Predictions
    y_pred = model.predict(X_processed)
    y_pred_proba = model.predict_proba(X_processed)
    
    st.markdown("---")
    
    # EVALUATION METRICS
    st.markdown("### Evaluation Metrics")
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    mcc = matthews_corrcoef(y, y_pred)
    
    try:
        auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    # Display metrics in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ACCURACY</div>
            <div class="metric-value">{accuracy:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">AUC SCORE</div>
            <div class="metric-value">{auc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label">PRECISION</div>
            <div class="metric-value">{precision:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-label">RECALL</div>
            <div class="metric-value">{recall:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="metric-label">F1 SCORE</div>
            <div class="metric-value">{f1:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">
            <div class="metric-label">MCC SCORE</div>
            <div class="metric-value">{mcc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CONFUSION MATRIX AND CLASSIFICATION REPORT
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Confusion Matrix")
        
        cm = confusion_matrix(y, y_pred)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Low (0)', 'Medium (1)', 'High (2)', 'Very High (3)'],
            y=['Low (0)', 'Medium (1)', 'High (2)', 'Very High (3)'],
            colorscale='Blues',
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 16},
            showscale=True,
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            height=400,
            font=dict(size=11),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Classification Report")
        
        # Get classification report as dict for better formatting
        from sklearn.metrics import classification_report
        report_dict = classification_report(y, y_pred, output_dict=True)
        
        # Create a clean dataframe
        report_data = []
        for label in ['0', '1', '2', '3']:
            if label in report_dict:
                report_data.append({
                    'Class': label,
                    'Precision': f"{report_dict[label]['precision']:.2f}",
                    'Recall': f"{report_dict[label]['recall']:.2f}",
                    'F1-Score': f"{report_dict[label]['f1-score']:.2f}",
                    'Support': int(report_dict[label]['support'])
                })
        
        # Add averages
        report_data.append({
            'Class': 'Accuracy',
            'Precision': '',
            'Recall': '',
            'F1-Score': f"{report_dict['accuracy']:.2f}",
            'Support': int(report_dict['macro avg']['support'])
        })
        report_data.append({
            'Class': 'Macro Avg',
            'Precision': f"{report_dict['macro avg']['precision']:.2f}",
            'Recall': f"{report_dict['macro avg']['recall']:.2f}",
            'F1-Score': f"{report_dict['macro avg']['f1-score']:.2f}",
            'Support': int(report_dict['macro avg']['support'])
        })
        report_data.append({
            'Class': 'Weighted Avg',
            'Precision': f"{report_dict['weighted avg']['precision']:.2f}",
            'Recall': f"{report_dict['weighted avg']['recall']:.2f}",
            'F1-Score': f"{report_dict['weighted avg']['f1-score']:.2f}",
            'Support': int(report_dict['weighted avg']['support'])
        })
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Prediction Analysis - REORGANIZED
    st.markdown("### Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Predicted distribution
        labels = ['Low (0)', 'Medium (1)', 'High (2)', 'Very High (3)']
        
        actual_counts = [len(y[y == i]) for i in range(4)]
        pred_counts = [len(y_pred[y_pred == i]) for i in range(4)]
        
        comparison_df = pd.DataFrame({
            'Price Range': labels * 2,
            'Count': actual_counts + pred_counts,
            'Type': ['Actual'] * 4 + ['Predicted'] * 4
        })
        
        fig = px.bar(
            comparison_df,
            x='Price Range',
            y='Count',
            color='Type',
            barmode='group',
            title='Actual vs Predicted Distribution',
            color_discrete_map={'Actual': '#667eea', 'Predicted': '#764ba2'}
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Download Results section - moved here
        st.markdown("### Download Results")
        
        # Summary stats
        correct = (y == y_pred).sum()
        total = len(y)
        
        st.markdown(f'<div class="info-box"><b>Prediction Summary:</b> {correct} out of {total} correct ({correct/total*100:.2f}%)</div>', unsafe_allow_html=True)
        
        # Create results dataframe
        max_proba = y_pred_proba.max(axis=1)
        results_df = df.copy()
        results_df['predicted_price_range'] = y_pred
        results_df['prediction_confidence'] = max_proba
        results_df['is_correct'] = y == y_pred
        
        csv = results_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("")
        st.markdown("")
        
        # Additional info
        st.markdown("**Download includes:**")
        st.markdown("- All original features")
        st.markdown("- Predicted price range")
        st.markdown("- Prediction confidence")
        st.markdown("- Correctness indicator")