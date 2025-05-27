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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Conv2D, 
                                    LSTM, Flatten, MaxPooling1D, MaxPooling2D,
                                    GlobalAveragePooling1D, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

class ModelTrainingTab:
    def render(self):
        st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
        
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
        
        # Hyperparameter Configuration
        st.subheader("Hyperparameter Configuration")
        
        hyperparams = {}
        
        # Traditional ML Models hyperparameters
        if model_category == "Traditional ML Models":
            if model_type == "Logistic Regression":
                col1, col2 = st.columns(2)
                with col1:
                    hyperparams['solver'] = st.selectbox("Solver", ["liblinear", "lbfgs", "newton-cg", "sag", "saga"])
                    hyperparams['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000, 100)
                with col2:
                    hyperparams['C'] = st.number_input("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.1)
                    hyperparams['penalty'] = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
            
            elif model_type == "Random Forest":
                col1, col2 = st.columns(2)
                with col1:
                    hyperparams['n_estimators'] = st.slider("Number of Trees", 10, 500, 100, 10)
                    hyperparams['criterion'] = st.selectbox(
                        "Criterion", 
                        ["gini", "entropy"] if problem_type == "Classification" else ["squared_error", "absolute_error"]
                    )
                with col2:
                    hyperparams['max_depth'] = st.slider("Max Depth", 2, 50, 10, 1)
                    hyperparams['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2, 1)
            
            elif model_type in ["SVM", "SVR"]:
                col1, col2 = st.columns(2)
                with col1:
                    hyperparams['kernel'] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                    hyperparams['C'] = st.number_input("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
                with col2:
                    hyperparams['degree'] = st.slider("Degree (for poly kernel)", 1, 10, 3, 1)
                    hyperparams['gamma'] = st.selectbox("Gamma", ["scale", "auto"])
            
            elif model_type == "Linear Regression":
                hyperparams['fit_intercept'] = st.checkbox("Fit Intercept", True)
            
            elif model_type == "Ridge Regression":
                hyperparams['alpha'] = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0, 0.01)
                hyperparams['solver'] = st.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
            
            elif model_type == "Lasso Regression":
                hyperparams['alpha'] = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0, 0.01)
                hyperparams['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000, 100)
        
        # Neural Network hyperparameters
        else:
            # Common hyperparameters
            col1, col2 = st.columns(2)
            with col1:
                hyperparams['epochs'] = st.slider("Epochs", 10, 500, 50, 10)
                hyperparams['batch_size'] = st.slider("Batch Size", 8, 256, 32, 8)
                hyperparams['optimizer'] = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
            
            with col2:
                hyperparams['learning_rate'] = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%f")
                hyperparams['activation'] = st.selectbox("Activation Function", ["relu", "tanh", "sigmoid", "elu", "selu"])
                hyperparams['early_stopping'] = st.checkbox("Early Stopping", True)
            
            # Network architecture
            st.subheader("Network Architecture")
            
            hyperparams['num_layers'] = st.slider("Number of Hidden Layers", 1, 5, 2, 1)
            
            # Layer configurations
            for i in range(hyperparams['num_layers']):
                st.markdown(f"#### Layer {i+1}")
                col1, col2 = st.columns(2)
                
                with col1:
                    layer_type_options = ["Dense"]
                    if model_type in ["CNN", "CNN Regressor"]:
                        layer_type_options = ["Conv1D", "Conv2D", "Dense"]
                    elif model_type in ["RNN/LSTM", "RNN/LSTM Regressor"]:
                        layer_type_options = ["LSTM", "Dense"]
                    
                    hyperparams[f'layer_{i}_type'] = st.selectbox(
                        f"Layer Type",
                        layer_type_options,
                        key=f"layer_{i}_type"
                    )
                
                with col2:
                    hyperparams[f'layer_{i}_units'] = st.slider(
                        f"Units/Filters",
                        16, 512, 64, 16,
                        key=f"layer_{i}_units"
                    )
                
                # Additional layer params based on type
                if hyperparams[f'layer_{i}_type'] in ["Conv1D", "Conv2D"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        hyperparams[f'layer_{i}_kernel'] = st.slider(
                            f"Kernel Size",
                            2, 10, 3, 1,
                            key=f"layer_{i}_kernel"
                        )
                    with col2:
                        hyperparams[f'layer_{i}_add_pooling'] = st.checkbox(
                            f"Add Pooling",
                            True,
                            key=f"layer_{i}_add_pooling"
                        )
                
                # Dropout option for any layer
                hyperparams[f'layer_{i}_dropout'] = st.slider(
                    f"Dropout Rate",
                    0.0, 0.5, 0.2, 0.1,
                    key=f"layer_{i}_dropout"
                )
        
        # Training Options
        st.subheader("Training Options")
        
        col1, col2 = st.columns(2)
        with col1:
            train_options = {}
            train_options['validation_split'] = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
            train_options['random_state'] = st.number_input("Random State", 0, 100, 42, 1)
        
        with col2:
            # Metrics to track
            if problem_type == "Classification":
                train_options['metrics'] = st.multiselect(
                    "Metrics to Track",
                    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                    default=["Accuracy", "F1 Score"]
                )
            else:
                train_options['metrics'] = st.multiselect(
                    "Metrics to Track",
                    ["RMSE", "MAE", "R² Score"],
                    default=["RMSE", "R² Score"]
                )
        
        # Train button
        if st.button("🚀 Start Training", use_container_width=True):
            # Extract train test data
            X_train = st.session_state.train_test_data['X_train']
            y_train = st.session_state.train_test_data['y_train']
            X_test = st.session_state.train_test_data['X_test']
            y_test = st.session_state.train_test_data['y_test']
            
            # Store selections in session state
            st.session_state.model_type = model_type
            st.session_state.selected_model_category = model_category
            
            # Train the model with progress bar
            with st.spinner("Training in progress..."):
                progress_bar = st.progress(0)
                
                # Build and train model
                model, history, train_time = self.train_model(
                    model_type, model_category, problem_type, 
                    hyperparams, X_train, y_train, progress_bar
                )
                
                # Make predictions
                if model_category == "Traditional ML Models":
                    y_pred = model.predict(X_test)
                    if problem_type == "Classification":
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)
                        else:
                            # For models without predict_proba, use decision_function if available
                            if hasattr(model, 'decision_function'):
                                decisions = model.decision_function(X_test)
                                if decisions.ndim == 1:  # Binary case
                                    y_proba = np.vstack([1-decisions, decisions]).T
                                else:  # Multi-class case
                                    y_proba = np.exp(decisions) / np.sum(np.exp(decisions), axis=1, keepdims=True)
                            else:
                                # Last resort: create dummy probabilities
                                n_classes = len(np.unique(y_test))
                                y_proba = np.zeros((len(y_test), n_classes))
                                for i, pred in enumerate(y_pred):
                                    y_proba[i, int(pred)] = 1
                    else:  # Regression
                        y_proba = None
                else:  # Neural Networks
                    if problem_type == "Classification":
                        y_proba = model.predict(X_test)
                        y_pred = np.argmax(y_proba, axis=1)
                    else:  # Regression
                        y_pred = model.predict(X_test).flatten()
                        y_proba = None
                
                # Calculate metrics
                if problem_type == "Classification":
                    metrics = self.calculate_classification_metrics(y_test, y_pred, y_proba)
                else:
                    metrics = self.calculate_regression_metrics(y_test, y_pred)
                
                # Store results in session state
                st.session_state.model = model
                st.session_state.experiment_results = {
                    "model_type": model_type,
                    "model_category": model_category,
                    "problem_type": problem_type,
                    "hyperparams": hyperparams,
                    "metrics": metrics,
                    "train_time": train_time,
                    "history": history,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "y_test": y_test
                }
            
            # Show success message
            st.success(f"✅ Model trained successfully in {train_time:.2f} seconds!")
            
            # Display key metrics
            st.subheader("Training Results")
            
            metric_cols = st.columns(len(metrics))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                with metric_cols[i]:
                    if isinstance(metric_value, float):
                        if metric_name in ["Accuracy"]:
                            st.metric(metric_name, f"{metric_value:.2%}")
                        else:
                            st.metric(metric_name, f"{metric_value:.4f}")
                    else:
                        st.metric(metric_name, str(metric_value))
            
            # Display training history for neural networks
            if model_category == "Neural Networks" and history:
                st.subheader("Training History")
                
                fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                
                # Plot training & validation loss
                axs[0].plot(history.history['loss'], label='Training Loss')
                if 'val_loss' in history.history:
                    axs[0].plot(history.history['val_loss'], label='Validation Loss')
                axs[0].set_title('Model Loss')
                axs[0].set_xlabel('Epoch')
                axs[0].set_ylabel('Loss')
                axs[0].legend(loc='upper right')
                
                # Plot accuracy for classification
                if problem_type == "Classification" and 'accuracy' in history.history:
                    axs[1].plot(history.history['accuracy'], label='Training Accuracy')
                    if 'val_accuracy' in history.history:
                        axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
                    axs[1].set_title('Model Accuracy')
                    axs[1].set_xlabel('Epoch')
                    axs[1].set_ylabel('Accuracy')
                    axs[1].legend(loc='lower right')
                
                st.pyplot(fig)
            
            # Option to save model
            st.subheader("Save Model")
            
            model_name = st.text_input("Model Name", f"{model_type.replace(' ', '_').lower()}_model")
            
            if st.button("Save Model"):
                # Make sure data directory exists
                os.makedirs("data", exist_ok=True)
                
                # Save model
                model_path = os.path.join("data", f"{model_name}.pkl")
                
                try:
                    if model_category == "Neural Networks":
                        # For neural networks, save in Keras format
                        keras_path = os.path.join("data", f"{model_name}")
                        model.save(keras_path)
                        
                        # Save experiment info separately
                        with open(os.path.join("data", f"{model_name}_info.pkl"), 'wb') as f:
                            pickle.dump({
                                "model_type": model_type,
                                "model_category": model_category,
                                "problem_type": problem_type,
                                "hyperparams": hyperparams,
                                "metrics": metrics,
                                "feature_columns": st.session_state.feature_columns,
                                "target_column": st.session_state.target_column
                            }, f)
                        
                        st.success(f"✅ Neural network model saved to {keras_path}")
                    else:
                        # For traditional ML models, use pickle
                        with open(model_path, 'wb') as f:
                            pickle.dump({
                                "model": model,
                                "model_type": model_type,
                                "model_category": model_category,
                                "problem_type": problem_type,
                                "hyperparams": hyperparams,
                                "metrics": metrics,
                                "feature_columns": st.session_state.feature_columns,
                                "target_column": st.session_state.target_column
                            }, f)
                        
                        st.success(f"✅ Model saved to {model_path}")
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
    
    def train_model(self, model_type, model_category, problem_type, hyperparams, X_train, y_train, progress_bar=None):
        start_time = time.time()
        history = None
        
        # Traditional ML Models
        if model_category == "Traditional ML Models":
            if model_type == "Logistic Regression":
                model = LogisticRegression(
                    solver=hyperparams.get('solver', 'liblinear'),
                    C=hyperparams.get('C', 1.0),
                    max_iter=hyperparams.get('max_iter', 1000),
                    penalty=hyperparams.get('penalty', 'l2'),
                    random_state=42
                )
            
            elif model_type == "Random Forest" and problem_type == "Classification":
                model = RandomForestClassifier(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 10),
                    criterion=hyperparams.get('criterion', 'gini'),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    random_state=42
                )
            
            elif model_type == "Random Forest" and problem_type == "Regression":
                model = RandomForestRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 10),
                    criterion=hyperparams.get('criterion', 'squared_error'),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    random_state=42
                )
            
            elif model_type == "SVM":
                model = SVC(
                    kernel=hyperparams.get('kernel', 'rbf'),
                    C=hyperparams.get('C', 1.0),
                    degree=hyperparams.get('degree', 3),
                    gamma=hyperparams.get('gamma', 'scale'),
                    probability=True,
                    random_state=42
                )
            
            elif model_type == "SVR":
                model = SVR(
                    kernel=hyperparams.get('kernel', 'rbf'),
                    C=hyperparams.get('C', 1.0),
                    degree=hyperparams.get('degree', 3),
                    gamma=hyperparams.get('gamma', 'scale')
                )
            
            elif model_type == "Linear Regression":
                model = LinearRegression(
                    fit_intercept=hyperparams.get('fit_intercept', True)
                )
            
            elif model_type == "Ridge Regression":
                model = Ridge(
                    alpha=hyperparams.get('alpha', 1.0),
                    solver=hyperparams.get('solver', 'auto'),
                    random_state=42
                )
            
            elif model_type == "Lasso Regression":
                model = Lasso(
                    alpha=hyperparams.get('alpha', 1.0),
                    max_iter=hyperparams.get('max_iter', 1000),
                    random_state=42
                )
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Update progress
            if progress_bar:
                progress_bar.progress(1.0)
        
        # Neural Networks
        else:
            # Get input shape
            input_shape = X_train.shape[1:]
            
            # Make sure input is reshaped correctly for different network types
            if model_type in ["CNN", "CNN Regressor"]:
                # Reshape for CNN if needed (assume 1D input)
                if len(input_shape) == 1:
                    # Reshape to (samples, timesteps, features) for Conv1D
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    input_shape = X_train.shape[1:]
            
            elif model_type in ["RNN/LSTM", "RNN/LSTM Regressor"]:
                # Reshape for LSTM if needed (assume 1D input)
                if len(input_shape) == 1:
                    # Reshape to (samples, timesteps, features)
                    # Use a sliding window approach if input is just a vector
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    input_shape = X_train.shape[1:]
            
            # Build model
            model = Sequential()
            
            # Add layers
            for i in range(hyperparams['num_layers']):
                layer_type = hyperparams.get(f'layer_{i}_type')
                units = hyperparams.get(f'layer_{i}_units', 64)
                activation = hyperparams.get('activation', 'relu')
                
                # First layer needs input_shape
                if i == 0:
                    if layer_type == 'Dense':
                        model.add(Dense(units, activation=activation, input_shape=input_shape))
                    elif layer_type == 'Conv1D':
                        kernel_size = hyperparams.get(f'layer_{i}_kernel', 3)
                        model.add(Conv1D(units, kernel_size, activation=activation, input_shape=input_shape))
                        if hyperparams.get(f'layer_{i}_add_pooling', True):
                            model.add(MaxPooling1D(2))
                    elif layer_type == 'Conv2D':
                        kernel_size = hyperparams.get(f'layer_{i}_kernel', 3)
                        model.add(Conv2D(units, kernel_size, activation=activation, input_shape=input_shape))
                        if hyperparams.get(f'layer_{i}_add_pooling', True):
                            model.add(MaxPooling2D(2))
                    elif layer_type == 'LSTM':
                        model.add(LSTM(units, activation=activation, input_shape=input_shape))
                else:
                    if layer_type == 'Dense':
                        model.add(Dense(units, activation=activation))
                    elif layer_type == 'Conv1D':
                        kernel_size = hyperparams.get(f'layer_{i}_kernel', 3)
                        model.add(Conv1D(units, kernel_size, activation=activation))
                        if hyperparams.get(f'layer_{i}_add_pooling', True):
                            model.add(MaxPooling1D(2))
                    elif layer_type == 'Conv2D':
                        kernel_size = hyperparams.get(f'layer_{i}_kernel', 3)
                        model.add(Conv2D(units, kernel_size, activation=activation))
                        if hyperparams.get(f'layer_{i}_add_pooling', True):
                            model.add(MaxPooling2D(2))
                    elif layer_type == 'LSTM':
                        model.add(LSTM(units, activation=activation))
                
                # Add dropout if specified
                dropout_rate = hyperparams.get(f'layer_{i}_dropout', 0.0)
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
            
            # Add flatten layer if needed
            if any(hyperparams.get(f'layer_{i}_type') in ['Conv1D', 'Conv2D', 'LSTM'] for i in range(hyperparams['num_layers'])):
                # Check if the last layer is not already flattened
                if hyperparams.get(f'layer_{hyperparams["num_layers"]-1}_type') not in ['Dense']:
                    model.add(Flatten())
            
            # Add output layer
            if problem_type == "Classification":
                # Get the number of classes
                n_classes = len(np.unique(y_train))
                if n_classes == 2:
                    model.add(Dense(1, activation='sigmoid'))
                    loss = 'binary_crossentropy'
                    # Convert y to binary format
                    y_train = y_train.astype(int)
                else:
                    model.add(Dense(n_classes, activation='softmax'))
                    loss = 'sparse_categorical_crossentropy'
                    # If y is already one-hot encoded, use categorical_crossentropy
                    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                        loss = 'categorical_crossentropy'
            else:
                model.add(Dense(1, activation='linear'))
                loss = 'mse'
            
            # Compile model
            if hyperparams.get('optimizer') == 'adam':
                optimizer = Adam(learning_rate=hyperparams.get('learning_rate', 0.001))
            elif hyperparams.get('optimizer') == 'sgd':
                optimizer = SGD(learning_rate=hyperparams.get('learning_rate', 0.001))
            else:
                optimizer = RMSprop(learning_rate=hyperparams.get('learning_rate', 0.001))
            
            model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=['accuracy'] if problem_type == "Classification" else ['mae']
            )
            
            # Set up callbacks
            callbacks = []
            if hyperparams.get('early_stopping', True):
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Train model with validation split
            epochs = hyperparams.get('epochs', 50)
            batch_size = hyperparams.get('batch_size', 32)
            
            # Custom callback for progress bar
            class ProgressBarCallback(tf.keras.callbacks.Callback):
                def __init__(self, epochs, progress_bar):
                    self.epochs = epochs
                    self.progress_bar = progress_bar
                
                def on_epoch_end(self, epoch, logs=None):
                    if self.progress_bar:
                        self.progress_bar.progress((epoch + 1) / self.epochs)
            
            if progress_bar:
                callbacks.append(ProgressBarCallback(epochs, progress_bar))
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
        
        train_time = time.time() - start_time
        
        return model, history, train_time
    
    def calculate_classification_metrics(self, y_true, y_pred, y_proba=None):
        # Convert to flat arrays to handle different formats
        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()
        
        metrics = {}
        
        # Basic metrics
        metrics['Accuracy'] = accuracy_score(y_true_flat, y_pred_flat)
        
        try:
            metrics['Precision'] = precision_score(y_true_flat, y_pred_flat, average='weighted')
            metrics['Recall'] = recall_score(y_true_flat, y_pred_flat, average='weighted')
            metrics['F1 Score'] = f1_score(y_true_flat, y_pred_flat, average='weighted')
        except Exception as e:
            metrics['Precision'] = "N/A"
            metrics['Recall'] = "N/A"
            metrics['F1 Score'] = "N/A"
        
        # ROC AUC if probabilities are available
        if y_proba is not None:
            try:
                # Handle different shapes of probability arrays
                if len(y_proba.shape) == 1 or y_proba.shape[1] == 1:
                    # Binary classification with single probability column
                    metrics['ROC AUC'] = roc_auc_score(y_true_flat, y_proba.flatten())
                elif y_proba.shape[1] == 2:
                    # Binary classification with two probability columns
                    metrics['ROC AUC'] = roc_auc_score(y_true_flat, y_proba[:, 1])
                else:
                    # Multi-class
                    metrics['ROC AUC'] = roc_auc_score(y_true_flat, y_proba, multi_class='ovr')
            except Exception as e:
                metrics['ROC AUC'] = "N/A"
        
        return metrics
    
    def calculate_regression_metrics(self, y_true, y_pred):
        # Convert to flat arrays
        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()
        
        metrics = {}
        
        # Calculate metrics
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        metrics['MAE'] = mean_absolute_error(y_true_flat, y_pred_flat)
        metrics['R² Score'] = r2_score(y_true_flat, y_pred_flat)
        
        return metrics 