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

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

# ---------------------------------
# Custom CSS for Beautiful UI
# ---------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stSelectbox label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# Paths
# ---------------------------------
MODEL_DIR = "models"
DATA_DIR = "data"

# ---------------------------------
# Load scaler
# ---------------------------------
@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

scaler = load_scaler()

# ---------------------------------
# Title
# ---------------------------------
st.markdown('<h1 class="main-title">📱 Mobile Price Classification</h1>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/smartphone.png", width=100)
    st.markdown("###  Select Model")
    
    model_name = st.selectbox(
        "Choose a classification model:",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )
    
    st.markdown("---")
    st.markdown("###  Dataset Info")
    st.info("""
    **Features**: 20  
    **Classes**: 4 price ranges  
    - 0: Low Cost
    - 1: Medium Cost
    - 2: High Cost
    - 3: Very High Cost
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Model Performance")
    
    # Performance preview based on model selected
    perf = {
        "Logistic Regression": {"acc": "96.50%", "f1": "0.965"},
        "Decision Tree": {"acc": "83.00%", "f1": "0.832"},
        "KNN": {"acc": "50.00%", "f1": "0.505"},
        "Naive Bayes": {"acc": "81.00%", "f1": "0.810"},
        "Random Forest": {"acc": "83.50%", "f1": "0.830"},
        "XGBoost": {"acc": "92.25%", "f1": "0.922"}
    }
    
    st.metric("Expected Accuracy", perf[model_name]["acc"])
    st.metric("Expected F1 Score", perf[model_name]["f1"])

# ---------------------------------
# Model Files Mapping
# ---------------------------------
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# ---------------------------------
# Load selected model
# ---------------------------------
@st.cache_resource
def load_model(model_file):
    return joblib.load(os.path.join(MODEL_DIR, model_file))

model = load_model(model_files[model_name])

# ---------------------------------
# Main Content
# ---------------------------------
st.markdown(f"### 🤖 Using Model: **{model_name}**")

# ---------------------------------
# Upload test CSV
# ---------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload test CSV (with price_range column)",
    type=["csv"],
    help="Upload your test data with mobile specifications and price_range column"
)

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.success(f" Data loaded successfully! **{len(df)} samples**")
    
    # Show data preview
    with st.expander("👁️ View Data Preview (First 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Prepare data
    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    
    # Scale only when required (matching your implementation)
    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_processed = scaler.transform(X)
    else:
        X_processed = X
    
    # Predictions
    y_pred = model.predict(X_processed)
    y_pred_proba = model.predict_proba(X_processed)
    
    st.markdown("---")
    
    # ---------------------------------
    # EVALUATION METRICS (Required!)
    # ---------------------------------
    st.markdown("###  Evaluation Metrics")
    
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
    
    # Display metrics in beautiful cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"> Accuracy</div>
            <div class="metric-value">{accuracy:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label"> AUC Score</div>
            <div class="metric-value">{auc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label"> Precision</div>
            <div class="metric-value">{precision:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-label"> Recall</div>
            <div class="metric-value">{recall:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="metric-label">⚡ F1 Score</div>
            <div class="metric-value">{f1:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">
            <div class="metric-label">🔢 MCC Score</div>
            <div class="metric-value">{mcc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---------------------------------
    # CONFUSION MATRIX (Required Visual!)
    # ---------------------------------
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔢 Confusion Matrix")
        
        cm = confusion_matrix(y, y_pred)
        
        # Create beautiful heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Low (0)', 'Medium (1)', 'High (2)', 'Very High (3)'],
            y=['Low (0)', 'Medium (1)', 'High (2)', 'Very High (3)'],
            colorscale='Blues',
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 18},
            showscale=True,
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Confusion Matrix Heatmap",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=450,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy per class
        st.markdown("#### 📊 Per-Class Accuracy")
        class_acc = []
        for i in range(4):
            mask = y == i
            if mask.sum() > 0:
                acc = accuracy_score(y[mask], y_pred[mask])
                class_acc.append({"Class": f"Class {i}", "Accuracy": f"{acc:.4f}"})
        
        st.dataframe(pd.DataFrame(class_acc), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("###  Classification Report")
        
        # Get report as text
        report_text = classification_report(y, y_pred)
        st.text(report_text)
        
        st.markdown("---")
        
        # Prediction confidence distribution
        st.markdown("####  Prediction Confidence")
        
        max_proba = y_pred_proba.max(axis=1)
        
        fig = px.histogram(
            x=max_proba,
            nbins=30,
            title='Confidence Distribution',
            labels={'x': 'Max Probability', 'y': 'Count'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.add_vline(
            x=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="50% Threshold"
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ---------------------------------
    # Additional Visualizations
    # ---------------------------------
    st.markdown("###  Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Predicted distribution
        labels = ['Low (0)', 'Med (1)', 'High (2)', 'V.High (3)']
        
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
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sample predictions
        st.markdown("#### 🎲 Sample Predictions")
        
        sample_size = min(10, len(df))
        price_map = {0: 'Low', 1: 'Med', 2: 'High', 3: 'V.High'}
        
        sample_df = pd.DataFrame({
            'Actual': [price_map[v] for v in y.iloc[:sample_size].values],
            'Predicted': [price_map[v] for v in y_pred[:sample_size]],
            'Confidence': [f"{max(p)*100:.1f}%" for p in y_pred_proba[:sample_size]],
            'Match': ['✅' if y.iloc[i] == y_pred[i] else '❌' for i in range(sample_size)]
        })
        
        # Color code the matches
        def highlight_match(row):
            if row['Match'] == '✅':
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)
        
        st.dataframe(
            sample_df.style.apply(highlight_match, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Summary stats
        correct = (y == y_pred).sum()
        total = len(y)
        
        st.info(f"**Summary**: {correct}/{total} correct predictions ({correct/total*100:.2f}%)")
    
    st.markdown("---")
    
    # ---------------------------------
    # Download Predictions
    # ---------------------------------
    st.markdown("###  Download Results")
    
    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_price_range'] = y_pred
    results_df['prediction_confidence'] = max_proba
    results_df['is_correct'] = y == y_pred
    
    csv = results_df.to_csv(index=False)
    
    st.download_button(
        label=" Download Predictions as CSV",
        data=csv,
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        help="Download all predictions with confidence scores"
    )

else:
    # Show instructions when no file uploaded
    st.info(" **Please upload a CSV file to begin analysis**")
    
    st.markdown("###  Expected Data Format")
    
    st.markdown("""
    Your CSV file should contain:
    - **20 feature columns**: battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi
    - **1 target column**: price_range (values: 0, 1, 2, or 3)
    """)
    
    # Example data
    example_data = pd.DataFrame({
        'battery_power': [842, 1021, 563],
        'blue': [0, 1, 1],
        'ram': [2549, 2631, 2603],
        'price_range': [1, 2, 2]
    })
    
    st.markdown("**Example:**")
    st.dataframe(example_data, use_container_width=True)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
        <strong>ML Assignment 2 - Mobile Price Classification</strong>
    </p>
    <p style='margin: 0;'>M.Tech (AIML/DSE) - BITS Pilani</p>
</div>
""", unsafe_allow_html=True)