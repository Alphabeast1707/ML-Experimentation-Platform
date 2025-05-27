import streamlit as st
from data_preprocessing import DataPreprocessingTab
from model_training import ModelTrainingTab
from model_evaluation import ModelEvaluationTab
from ai_assistant import AIAssistantTab

# Set page config
st.set_page_config(
    page_title="ML Experimentation Hub",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'train_test_data' not in st.session_state:
    st.session_state.train_test_data = None
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Main app title
st.title("🤖 ML Experimentation Hub")

# Create tabs
tabs = st.tabs([
    "Data Preprocessing",
    "Model Training",
    "Model Evaluation",
    "AI Assistant"
])

# Render each tab
with tabs[0]:
    data_preprocessing_tab = DataPreprocessingTab()
    data_preprocessing_tab.render()

with tabs[1]:
    model_training_tab = ModelTrainingTab()
    model_training_tab.render()

with tabs[2]:
    model_evaluation_tab = ModelEvaluationTab()
    model_evaluation_tab.render()

with tabs[3]:
    ai_assistant_tab = AIAssistantTab()
    ai_assistant_tab.render()

# Add custom CSS
st.markdown("""
<style>
    .sub-header {
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e1f5fe;
        border-left: 5px solid #03a9f4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True) 