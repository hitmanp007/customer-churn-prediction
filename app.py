
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource
def load_model_artifacts():
    """Load trained model and required artifacts"""
    try:
        # Adjust path based on your project structure
        model_path = Path(__file__).parent / "model" / "churn_model.pkl"
        columns_path = Path(__file__).parent / "model" / "model_columns.pkl"
        
        # Alternative path if files are in parent directory
        if not model_path.exists():
            model_path = Path(__file__).parent.parent / "model" / "churn_model.pkl"
            columns_path = Path(__file__).parent.parent / "model" / "model_columns.pkl"
        
        model = pickle.load(open(model_path, "rb"))
        model_columns = pickle.load(open(columns_path, "rb"))
        
        return model, model_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load sample data for visualizations
@st.cache_data
def load_real_data():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # 🔥 ADD THIS LINE (IMPORTANT FIX)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


    return df

# Sidebar navigation
st.sidebar.markdown("# 📊 Customer Churn Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🎯 Prediction", "📈 Dashboard", "🔍 Model Insights"],
    help="Select a page to navigate"
)

# Load model
model, model_columns = load_model_artifacts()

# Function to preprocess input data
def preprocess_input_data(input_dict, model_columns):
    
    # Step 1: Create empty dataframe with ALL columns
    input_df = pd.DataFrame(
        np.zeros((1, len(model_columns))),
        columns=model_columns
    )

    # Step 2: Fill numeric features
    input_df["tenure"] = input_dict["tenure"]
    input_df["MonthlyCharges"] = input_dict["MonthlyCharges"]
    input_df["TotalCharges"] = input_dict["TotalCharges"]

    # Step 3: SAFE categorical mapping (robust)
    # 🔥 EXACT MATCH (FIX)
    mappings = {
    "Contract": input_dict["contract"],
    "InternetService": input_dict["internet_service"],
    "OnlineSecurity": input_dict["online_security"],
    "TechSupport": input_dict["tech_support"],
    "PaymentMethod": input_dict["payment_method"],
    "PaperlessBilling": input_dict["paperless_billing"]
    }

    for key, value in mappings.items():
        col_name = f"{key}_{value}"
        if col_name in input_df.columns:
            input_df[col_name] = 1

    return input_df

# ==================== PREDICTION PAGE ====================
if page == "🎯 Prediction":
    st.markdown('<div class="main-header">🎯 Customer Churn Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("### Enter Customer Information")
    
    # Create two columns for input organization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Account Information")
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            help="Number of months the customer has been with the company"
        )
        
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=5.0,
            help="Monthly charges for services"
        )
        
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=2000.0,
            step=100.0,
            help="Total charges accumulated"
        )
        
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help="Type of contract with the customer"
        )
        
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            help="Whether customer uses paperless billing"
        )
    
    with col2:
        st.markdown("#### 🌐 Service Information")
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            help="Type of internet service"
        )
        
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"],
            help="Whether customer has online security add-on"
        )
        
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No"],
            help="Whether customer has tech support add-on"
        )
        
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            help="Customer's preferred payment method"
        )
    
    # Prediction button
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        if model is not None and model_columns is not None:
            # Prepare input data
            input_dict = {
                'tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'contract': contract,
                'internet_service': internet_service,
                'online_security': online_security,
                'tech_support': tech_support,
                'payment_method': payment_method,
                'paperless_billing': paperless_billing
            }
            
            # Preprocess input
            input_df = preprocess_input_data(input_dict, model_columns)

            # 🔥 ADD THIS LINE HERE
            st.write("Final Input to Model:", input_df.head())

            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            # Display results in a professional card
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Churn Probability",
                    value=f"{probability:.1%}",
                    delta=None
                )
            
            with col2:
                risk_level = "High" if probability > 0.5 else "Medium" if probability > 0.3 else "Low"
                st.metric(
                    label="Risk Level",
                    value=risk_level,
                    delta=None
                )
            
            with col3:
                recommendation = "Immediate action required" if probability > 0.5 else "Monitor closely" if probability > 0.3 else "Low priority"
                st.metric(
                    label="Recommendation",
                    value=recommendation,
                    delta=None
                )
            
            # Visualization of probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                title = {'text': "Churn Risk Score"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "darkred" if probability > 0.5 else "orange" if probability > 0.3 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "salmon"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100}
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display risk factors
            st.markdown("#### ⚠️ Key Risk Factors")
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("• Month-to-month contract (higher churn risk)")
            if internet_service == "Fiber optic":
                risk_factors.append("• Fiber optic service (higher churn rate)")
            if monthly_charges > 100:
                risk_factors.append(f"• High monthly charges (${monthly_charges:.2f})")
            if tenure < 12:
                risk_factors.append("• New customer (less than 1 year)")
            if payment_method == "Electronic check":
                risk_factors.append("• Electronic check payment (higher churn risk)")
            if online_security == "No":
                risk_factors.append("• No online security add-on")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.write("✓ No significant risk factors identified")
            
        else:
            st.error("Model not loaded properly. Please check model files.")

# ==================== DASHBOARD PAGE ====================
elif page == "📈 Dashboard":
    st.markdown('<div class="main-header">📈Customer churn ' \
    'Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Load sample datar
    df = load_real_data()
    
    # Key metrics
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}", help="Total number of customers")
    with col2:
        churn_rate = (df['Churn'].sum() / len(df)) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%", delta=None)
    with col3:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg. Tenure", f"{avg_tenure:.1f} months", delta=None)
    with col4:
        avg_charges = df['MonthlyCharges'].mean()
        st.metric("Avg. Monthly Charges", f"${avg_charges:.2f}", delta=None)
    
    # Churn by category visualizations
    st.markdown("---")
    st.markdown("### Churn Analysis by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Contract type vs churn
        contract_churn = df.groupby('Contract')['Churn'].mean() * 100
        fig = px.bar(
            x=contract_churn.index,
            y=contract_churn.values,
            title="Churn Rate by Contract Type",
            labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
            color=contract_churn.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Internet service vs churn
        internet_churn = df.groupby('InternetService')['Churn'].mean() * 100
        fig = px.bar(
            x=internet_churn.index,
            y=internet_churn.values,
            title="Churn Rate by Internet Service",
            labels={'x': 'Internet Service', 'y': 'Churn Rate (%)'},
            color=internet_churn.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payment method vs churn
        payment_churn = df.groupby('PaymentMethod')['Churn'].mean() * 100
        fig = px.bar(
            x=payment_churn.index,
            y=payment_churn.values,
            title="Churn Rate by Payment Method",
            labels={'x': 'Payment Method', 'y': 'Churn Rate (%)'},
            color=payment_churn.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tenure distribution by churn status
        fig = px.histogram(
            df,
            x='tenure',
            color='Churn',
            title="Tenure Distribution by Churn Status",
            labels={'tenure': 'Tenure (months)', 'count': 'Number of Customers'},
            color_discrete_map={0: 'green', 1: 'red'},
            barmode='overlay'
        )
        fig.update_layout(legend_title_text='Churned')
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly charges distribution
    st.markdown("---")
    st.markdown("### Monthly Charges Analysis")
    
    fig = px.box(
        df,
        x='Churn',
        y='MonthlyCharges',
        title="Monthly Charges Distribution by Churn Status",
        labels={'Churn': 'Churned', 'MonthlyCharges': 'Monthly Charges ($)'},
        color='Churn',
        color_discrete_map={0: 'green', 1: 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== MODEL INSIGHTS PAGE ====================
else:
    st.markdown('<div class="main-header">🔍 Model Insights</div>', unsafe_allow_html=True)
    
    st.markdown("### Model Performance & Feature Importance")
    
    # Create tabs for different insights
    tab1, tab2, tab3 = st.tabs(["📊 Model Metrics", "🎯 Feature Importance", "💡 Business Insights"])
    
    with tab1:
        st.markdown("#### Model Performance Metrics")
        
        # Display sample metrics (replace with actual model metrics if available)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", "82.3%", help="Overall prediction accuracy")
        with col2:
            st.metric("Precision", "79.5%", help="Precision for churn prediction")
        with col3:
            st.metric("Recall", "76.8%", help="Recall for churn prediction")
        
        # Confusion matrix visualization
        confusion_matrix = np.array([[450, 50], [80, 120]])
        
        fig = px.imshow(
            confusion_matrix,
            text_auto=True,
            x=['Predicted Non-Churn', 'Predicted Churn'],
            y=['Actual Non-Churn', 'Actual Churn'],
            title="Confusion Matrix",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve placeholder
        st.markdown("#### ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 0.2, 0.4, 0.6, 0.8, 1],
            y=[0, 0.6, 0.75, 0.85, 0.92, 1],
            mode='lines',
            name='ROC Curve (AUC = 0.89)',
            line=dict(color='darkblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Top Features Influencing Churn")
        
        # Sample feature importance (replace with actual model feature importance)
        features = ['Contract Type', 'Tenure', 'Monthly Charges', 'Internet Service', 
                'Payment Method', 'Tech Support', 'Online Security', 'Total Charges']
        importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Feature Importance",
            labels={'x': 'Importance Score', 'y': 'Features'},
            color=importance,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Feature Impact Analysis")
        st.info("""
        **Key Insights:**
        - **Contract Type** is the strongest predictor of churn
        - Customers with month-to-month contracts are 3x more likely to churn
        - **Tenure** shows that newer customers are at higher risk
        - High **Monthly Charges** (>$100) significantly increase churn probability
        - Lack of **Tech Support** and **Online Security** services correlates with higher churn
        """)
    
    with tab3:
        st.markdown("#### Business Recommendations")
        
        st.markdown("""
        ### 💡 Actionable Insights
        
        Based on the model analysis, here are key recommendations to reduce churn:
        
        #### 1. **Contract Management**
        - Offer incentives for longer-term contracts (1-2 year)
        - Implement automatic renewal reminders
        - Create loyalty programs for month-to-month customers
        
        #### 2. **Pricing Strategy**
        - Review pricing for high-risk segments
        - Offer personalized discounts for at-risk customers
        - Bundle services to increase value proposition
        
        #### 3. **Service Enhancement**
        - Promote add-on services (Tech Support, Online Security)
        - Improve fiber optic service quality
        - Provide proactive technical support
        
        #### 4. **Customer Engagement**
        - Early intervention for new customers (<6 months)
        - Regular check-ins for high-risk segments
        - Personalized retention offers based on churn probability
        
        #### 5. **Payment Optimization**
        - Encourage automatic payment methods
        - Offer discounts for bank transfer/credit card users
        - Simplify electronic check payment process
        """)
        
        # Add a retention strategy matrix
        st.markdown("### Retention Strategy Matrix")
        
        retention_data = pd.DataFrame({
            'Strategy': ['Early Engagement', 'Loyalty Programs', 'Personalized Offers', 'Service Upgrades'],
            'Impact': [8.5, 7.8, 9.2, 7.5],
            'Implementation Effort': [6.0, 7.0, 8.5, 5.5]
        })
        
        fig = px.scatter(
            retention_data,
            x='Implementation Effort',
            y='Impact',
            text='Strategy',
            size=[30, 30, 30, 30],
            title="Strategy Impact vs. Implementation Effort",
            labels={'Impact': 'Expected Impact (1-10)', 'Implementation Effort': 'Implementation Effort (1-10)'}
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by Machine Learning | Customer Churn Prediction Model</p>",
    unsafe_allow_html=True
)