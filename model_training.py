import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import os
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, r2_score,
                             mean_squared_error, mean_absolute_error)

# Conditionally import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Conv2D, 
                                        LSTM, Flatten, MaxPooling1D, MaxPooling2D,
                                        GlobalAveragePooling1D, GlobalAveragePooling2D)
    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    
class ModelTrainingTab:
    def render(self):
        st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
        
        # Check if TensorFlow is available
        if not 'TENSORFLOW_AVAILABLE' in globals() or not TENSORFLOW_AVAILABLE:
            st.warning("⚠️ TensorFlow is not available. Deep learning models will be disabled.")
            st.info("Only traditional ML models will be available for training.")
        
        # Check if data is preprocessed
        if st.session_state.train_test_data is None:
            st.warning("⚠️ Please complete data preprocessing first!")
            st.info("Go to the Data Preprocessing tab to prepare your data.")
            return
        
        # Problem Type
        problem_type = st.session_state.problem_type
        if not problem_type:
            st.warning("⚠️ Problem type not defined. Please set it in the Data Preprocessing tab.")
            return
        
        # Training Configuration
        st.markdown("### 🔧 Training Configuration")
        
        # Model Selection
        st.subheader("Select Model")
        
        # If TensorFlow is not available, only show traditional ML models
        if not 'TENSORFLOW_AVAILABLE' in globals() or not TENSORFLOW_AVAILABLE:
            model_category = "Traditional ML Models"
        else:
            model_category = st.radio(
                "Model Category",
                ["Traditional ML Models", "Neural Networks"],
                key="model_category"
            )
        
        if model_category == "Traditional ML Models":
            if problem_type == "Classification":
                model_options = {
                    "Logistic Regression": "Simple linear model for classification",
                    "Random Forest": "Ensemble of decision trees",
                    "SVM": "Support Vector Machine with various kernels"
                }
            else:  # Regression
                model_options = {
                    "Linear Regression": "Simple linear model for regression",
                    "Ridge Regression": "Linear regression with L2 regularization",
                    "Lasso Regression": "Linear regression with L1 regularization",
                    "Random Forest": "Ensemble of decision trees for regression",
                    "SVR": "Support Vector Regression"
                }
        else:  # Neural Networks
            if problem_type == "Classification":
                model_options = {
                    "Dense Neural Network": "Simple fully connected network",
                    "CNN": "Convolutional Neural Network (for structured data)",
                    "RNN/LSTM": "Recurrent Neural Network for sequential data"
                }
            else:  # Regression
                model_options = {
                    "Dense Neural Network": "Simple fully connected network for regression",
                    "CNN Regressor": "Convolutional Neural Network for structured data",
                    "RNN/LSTM Regressor": "Recurrent Neural Network for sequential data"
                }
        
        # Display options with descriptions
        model_type = st.selectbox(
            "Select Model Type",
            list(model_options.keys()),
            format_func=lambda x: f"{x}: {model_options[x]}"
        )
