import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and preprocessor
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_artifacts()

# App title and description
st.title(" Telco Customer Churn Prediction")
st.markdown("""
Predict the likelihood of customer churn based on service details and demographics.
Adjust the parameters in the sidebar and click **Predict Churn** to see results.
""")

# Sidebar for user input
with st.sidebar:
    st.header("Customer Information")
    st.subheader("Demographics")
    
    # Demographic inputs
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col2:
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    st.subheader("Account Information")
    
    # Account information
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    st.subheader("Services")
    
    # Service information
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Internet-dependent services
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    st.subheader("Charges")
    
    # Charges
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
    
    # Prediction button
    predict_btn = st.button("Predict Churn", type="primary", use_container_width=True)

# Create input dictionary
input_data = {
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Display user input
st.subheader("Customer Details")
st.dataframe(input_df.T.rename(columns={0: "Value"}))

# Prediction and results
if predict_btn:
    # Preprocess input
    processed_input = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)[0]
    
    # Display results
    st.subheader("Prediction Results")
    
    # Create columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction card
        if prediction[0] == 1:
            st.error(f" High Churn Risk: {probability[1]*100:.1f}% probability")
        else:
            st.success(f" Low Churn Risk: {probability[0]*100:.1f}% probability")
        
        # Probability gauge
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['Churn Probability'], [probability[1]], color='#ff4b4b' if probability[1] > 0.5 else '#2ecc71')
        ax.set_xlim(0, 1)
        ax.set_title('Churn Probability Score')
        ax.text(probability[1] + 0.02, 0, f'{probability[1]*100:.1f}%', 
                va='center', fontsize=12, color='black')
        st.pyplot(fig)
    
    with col2:
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("**Top Influencing Factors**")
            feature_importances = model.feature_importances_
            feature_names = preprocessor.get_feature_names_out()
            
            # Get top 5 features
            top_features_idx = np.argsort(feature_importances)[::-1][:5]
            top_features = feature_names[top_features_idx]
            top_importance = feature_importances[top_features_idx]
            
            # Display as bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top_features, top_importance, color='#3498db')
            ax.set_title('Top Factors Influencing Prediction')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model")
    
    # Interpretation and recommendations
    st.subheader("Recommendations")
    if prediction[0] == 1:
        st.warning("""
        **This customer is at high risk of churning!**  
        Recommended actions:
        - Offer personalized retention incentives
        - Assign to a dedicated account manager
        - Analyze service pain points
        - Proactively reach out with special offers
        """)
    else:
        st.info("""
        **This customer has low churn risk.**  
        Recommended actions:
        - Continue providing excellent service
        - Monitor for any changes in usage patterns
        - Consider upselling opportunities
        - Maintain regular engagement
        """)

# Footer
st.divider()
st.markdown("""
**About this App**:  
This predictive model was built using the Telco Customer Churn dataset from Kaggle.
The machine learning model helps identify customers at risk of leaving the service.
""")