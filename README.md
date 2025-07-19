#  Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge)


A predictive analytics solution that identifies customers at risk of leaving telecom services, enabling proactive retention strategies.

![Churn Prediction App Screenshot](https://via.placeholder.com/800x400?text=App+Screenshot+Here) *(Replace with actual screenshot)*

## 🔍 Project Overview
This end-to-end machine learning project:
- Processes customer service data from the [Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Identifies key churn indicators (contract type, tenure, services used)
- Predicts churn probability with **85%+ recall** (minimizing false negatives)
- Deploys an interactive web app for business teams

**Business Impact**: Potential to reduce churn by 20-30% through targeted interventions.

## 🚀 Key Features
| Feature | Description |
|---------|-------------|
| **Data Pipeline** | Automated preprocessing (missing values, feature engineering) |
| **Model Comparison** | Tested Logistic Regression, Random Forest, XGBoost |
| **Performance Metrics** | Precision, Recall, ROC-AUC, SHAP explainability |
| **Production App** | Streamlit interface for real-time predictions |
| **Deployment Ready** | Packaged with Docker support |





## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone repository
git clone https://github.com/EzatHotak/telco-churn-prediction.git
cd telco-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py