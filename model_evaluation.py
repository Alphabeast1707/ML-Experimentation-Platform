import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, precision_recall_curve, auc)
import os
import pickle
import json

class ModelEvaluationTab:
    def render(self):
        st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)
        
        # Check if a model has been trained
        if st.session_state.experiment_results is None:
            st.warning("⚠️ No model has been trained yet!")
            st.info("Please go to the Model Training tab to train a model first.")
            return
        
        # Get experiment results
        results = st.session_state.experiment_results
        problem_type = results["problem_type"]
        model_type = results["model_type"]
        model_category = results["model_category"]
        metrics = results["metrics"]
        
        # Display model information
        st.markdown("### 🤖 Model Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", model_type)
        with col2:
            st.metric("Category", model_category)
        with col3:
            st.metric("Problem Type", problem_type)
        
        # Display metrics
        st.markdown("### 📊 Performance Metrics")
        
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
        
        # Evaluation tabs based on problem type
        if problem_type == "Classification":
            self.classification_evaluation(results)
        else:
            self.regression_evaluation(results)
        
        # Model comparisons
        st.markdown("### 🔍 Model Comparison")
        
        # Check if there are saved models to compare
        data_dir = os.path.join("data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Find all saved models
        saved_models = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and not f.endswith('_info.pkl')]
        saved_models += [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f != "__pycache__"]
        
        if not saved_models:
            st.info("No saved models found for comparison. Save your models after training to compare them.")
        else:
            # Allow user to select models to compare
            models_to_compare = st.multiselect(
                "Select models to compare",
                saved_models,
                default=[]
            )
            
            if models_to_compare and st.button("Compare Models"):
                self.compare_models(models_to_compare, problem_type)
    
    def classification_evaluation(self, results):
        st.markdown("### 📈 Classification Analysis")
        
        # Extract data
        y_true = results["y_test"]
        y_pred = results["y_pred"]
        y_proba = results.get("y_proba")
        
        # Convert to flat arrays
        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
        class_names = [str(c) for c in classes]
        
        # Evaluation tabs
        eval_tabs = st.tabs(["Confusion Matrix", "Classification Report", "ROC Curve", "Precision-Recall Curve"])
        
        # Confusion Matrix
        with eval_tabs[0]:
            st.subheader("Confusion Matrix")
            
            cm = confusion_matrix(y_true_flat, y_pred_flat)
            
            # Normalize confusion matrix
            normalize = st.checkbox("Normalize", value=True, key="normalize_cm")
            
            if normalize:
                # Normalize by row (true labels)
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.round(cm_norm * 100, 1)  # Convert to percentages
                cm_display = cm_norm
                fmt = '.1f'
                title = "Normalized Confusion Matrix (%)"
            else:
                cm_display = cm
                fmt = 'd'
                title = "Confusion Matrix (Counts)"
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                            xticklabels=class_names, yticklabels=class_names, ax=ax)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            st.pyplot(fig)
            
            # Add counts and percentages
            st.write("**Overall Accuracy:**", f"{results['metrics'].get('Accuracy', 0):.2%}")
            st.write(f"**Total Samples:** {np.sum(cm)}")
            st.write(f"**Correct Predictions:** {np.trace(cm)} ({np.trace(cm)/np.sum(cm):.2%})")
            st.write(f"**Incorrect Predictions:** {np.sum(cm) - np.trace(cm)} ({(np.sum(cm) - np.trace(cm))/np.sum(cm):.2%})")
        
        # Classification Report
        with eval_tabs[1]:
            st.subheader("Classification Report")
            
            # Generate classification report
            cr = classification_report(y_true_flat, y_pred_flat, output_dict=True)
            
            # Convert to DataFrame
            cr_df = pd.DataFrame(cr).transpose()
            
            # Display as a table
            st.dataframe(cr_df.style.format("{:.4f}"), use_container_width=True)
            
            # Class Distribution
            st.subheader("Class Distribution")
            
            # Count true and predicted values
            true_counts = pd.Series(y_true_flat).value_counts().sort_index()
            pred_counts = pd.Series(y_pred_flat).value_counts().sort_index()
            
            # Combine into a DataFrame
            dist_df = pd.DataFrame({
                'True': pd.Series(true_counts, index=classes),
                'Predicted': pd.Series(pred_counts, index=classes)
            }).fillna(0).astype(int)
            
            # Calculate percentages
            dist_df['True %'] = dist_df['True'] / dist_df['True'].sum() * 100
            dist_df['Predicted %'] = dist_df['Predicted'] / dist_df['Predicted'].sum() * 100
            
            # Display table
            st.dataframe(dist_df, use_container_width=True)
            
            # Visualize distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Bar chart
            x = np.arange(len(classes))
            width = 0.35
            
            ax.bar(x - width/2, dist_df['True'], width, label='True')
            ax.bar(x + width/2, dist_df['Predicted'], width, label='Predicted')
            
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution: True vs Predicted')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names)
            ax.legend()
            
            st.pyplot(fig)
        
        # ROC Curve
        with eval_tabs[2]:
            st.subheader("ROC Curve")
            
            if y_proba is None:
                st.warning("Probability estimates not available for this model.")
            else:
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Binary Classification
                    if len(classes) == 2:
                        # Extract probability of the positive class
                        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                            y_prob = y_proba[:, 1]
                        else:
                            y_prob = y_proba.flatten()
                        
                        # Compute ROC curve and ROC area
                        fpr, tpr, _ = roc_curve(y_true_flat, y_prob)
                        roc_auc = auc(fpr, tpr)
                        
                        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic (ROC) Curve')
                        plt.legend(loc="lower right")
                    
                    # Multiclass Classification
                    else:
                        for i, cls in enumerate(classes):
                            # Create binary labels for one-vs-rest
                            binary_y_true = (y_true_flat == cls).astype(int)
                            
                            # Get probability for the current class
                            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                                y_prob = y_proba[:, i]
                            else:
                                # If only one probability is available, use it for class 1 only
                                if cls == 1:
                                    y_prob = y_proba.flatten()
                                else:
                                    y_prob = 1 - y_proba.flatten()
                            
                            # Compute ROC curve and ROC area
                            fpr, tpr, _ = roc_curve(binary_y_true, y_prob)
                            roc_auc = auc(fpr, tpr)
                            
                            plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (area = {roc_auc:.2f})')
                        
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic (ROC) Curve - One vs Rest')
                        plt.legend(loc="lower right")
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating ROC curve: {str(e)}")
        
        # Precision-Recall Curve
        with eval_tabs[3]:
            st.subheader("Precision-Recall Curve")
            
            if y_proba is None:
                st.warning("Probability estimates not available for this model.")
            else:
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Binary Classification
                    if len(classes) == 2:
                        # Extract probability of the positive class
                        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                            y_prob = y_proba[:, 1]
                        else:
                            y_prob = y_proba.flatten()
                        
                        # Compute Precision-Recall curve
                        precision, recall, _ = precision_recall_curve(y_true_flat, y_prob)
                        pr_auc = auc(recall, precision)
                        
                        plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title('Precision-Recall Curve')
                        plt.legend(loc="lower left")
                    
                    # Multiclass Classification
                    else:
                        for i, cls in enumerate(classes):
                            # Create binary labels for one-vs-rest
                            binary_y_true = (y_true_flat == cls).astype(int)
                            
                            # Get probability for the current class
                            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                                y_prob = y_proba[:, i]
                            else:
                                # If only one probability is available, use it for class 1 only
                                if cls == 1:
                                    y_prob = y_proba.flatten()
                                else:
                                    y_prob = 1 - y_proba.flatten()
                            
                            # Compute Precision-Recall curve
                            precision, recall, _ = precision_recall_curve(binary_y_true, y_prob)
                            pr_auc = auc(recall, precision)
                            
                            plt.plot(recall, precision, lw=2, label=f'Class {cls} (area = {pr_auc:.2f})')
                        
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title('Precision-Recall Curve - One vs Rest')
                        plt.legend(loc="lower left")
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating Precision-Recall curve: {str(e)}")
    
    def regression_evaluation(self, results):
        st.markdown("### 📈 Regression Analysis")
        
        # Extract data
        y_true = results["y_test"]
        y_pred = results["y_pred"]
        
        # Convert to flat arrays
        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()
        
        # Evaluation tabs
        eval_tabs = st.tabs(["Prediction Plot", "Residual Analysis", "Error Distribution"])
        
        # Prediction Plot
        with eval_tabs[0]:
            st.subheader("Actual vs Predicted Values")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot
            ax.scatter(y_true_flat, y_pred_flat, alpha=0.5)
            
            # Perfect prediction line
            min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
            max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values')
            
            # Add R² to the plot
            r2 = results['metrics'].get('R² Score', 0)
            ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                      fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            st.pyplot(fig)
            
            # Data table with predictions
            st.subheader("Prediction Details")
            
            # Create a DataFrame with true and predicted values
            pred_df = pd.DataFrame({
                'Actual': y_true_flat,
                'Predicted': y_pred_flat,
                'Absolute Error': np.abs(y_true_flat - y_pred_flat),
                'Squared Error': (y_true_flat - y_pred_flat) ** 2
            })
            
            # Add percent error column for non-zero actual values
            pred_df['Percent Error'] = np.where(
                y_true_flat != 0,
                np.abs((y_true_flat - y_pred_flat) / y_true_flat) * 100,
                np.nan
            )
            
            # Show the top errors
            st.write("Top 10 largest errors:")
            st.dataframe(pred_df.sort_values('Absolute Error', ascending=False).head(10))
        
        # Residual Analysis
        with eval_tabs[1]:
            st.subheader("Residual Analysis")
            
            # Calculate residuals
            residuals = y_true_flat - y_pred_flat
            
            # Residuals vs predicted values
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(y_pred_flat, residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Predicted Values')
            
            st.pyplot(fig)
            
            # Residuals vs features if available
            if hasattr(st.session_state, 'feature_columns') and st.session_state.feature_columns:
                feature_columns = st.session_state.feature_columns
                
                # Get X_test data from the train test data
                if hasattr(st.session_state, 'train_test_data') and 'X_test' in st.session_state.train_test_data:
                    X_test = st.session_state.train_test_data['X_test']
                    
                    # Allow user to select a feature for residual analysis
                    selected_feature = st.selectbox(
                        "Select a feature for residual analysis",
                        feature_columns
                    )
                    
                    # Get the feature values
                    if isinstance(X_test, pd.DataFrame):
                        feature_values = X_test[selected_feature].values
                    else:
                        # If X_test is not a DataFrame, we need to know which column is which
                        feature_idx = feature_columns.index(selected_feature)
                        feature_values = X_test[:, feature_idx]
                    
                    # Plot residuals vs the selected feature
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.scatter(feature_values, residuals, alpha=0.5)
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_xlabel(selected_feature)
                    ax.set_ylabel('Residuals')
                    ax.set_title(f'Residuals vs {selected_feature}')
                    
                    st.pyplot(fig)
        
        # Error Distribution
        with eval_tabs[2]:
            st.subheader("Error Distribution")
            
            # Calculate errors
            errors = y_true_flat - y_pred_flat
            
            # Histogram of errors
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.histplot(errors, kde=True, ax=ax)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Error')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Errors')
            
            st.pyplot(fig)
            
            # Statistical summary of errors
            st.subheader("Error Statistics")
            
            error_stats = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [
                    np.mean(errors),
                    np.median(errors),
                    np.std(errors),
                    np.min(errors),
                    np.percentile(errors, 25),
                    np.percentile(errors, 50),
                    np.percentile(errors, 75),
                    np.max(errors)
                ]
            })
            
            st.dataframe(error_stats, use_container_width=True)
    
    def compare_models(self, model_files, problem_type):
        st.subheader("Model Comparison")
        
        data_dir = os.path.join("data")
        
        # Load model data
        models_data = []
        
        for model_file in model_files:
            model_path = os.path.join(data_dir, model_file)
            
            try:
                # Check if it's a directory (Keras model) or file (pickle)
                if os.path.isdir(model_path):
                    # For keras models, look for the info file
                    info_path = os.path.join(data_dir, f"{model_file}_info.pkl")
                    if os.path.exists(info_path):
                        with open(info_path, 'rb') as f:
                            model_info = pickle.load(f)
                            models_data.append(model_info)
                else:
                    # For sklearn models
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        # Skip if it doesn't have the right structure
                        if isinstance(model_data, dict) and 'metrics' in model_data:
                            models_data.append(model_data)
            except Exception as e:
                st.error(f"Error loading model {model_file}: {str(e)}")
        
        if not models_data:
            st.warning("No valid model data could be loaded for comparison.")
            return
        
        # Create comparison table
        comparison_data = []
        
        for model_data in models_data:
            row = {
                'Model': model_data.get('model_type', 'Unknown'),
                'Category': model_data.get('model_category', 'Unknown'),
            }
            
            # Add metrics
            metrics = model_data.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    row[metric_name] = metric_value
            
            comparison_data.append(row)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Visualize comparison
        st.subheader("Metric Comparison Visualization")
        
        # Get metrics to visualize
        metrics_to_visualize = []
        if problem_type == "Classification":
            metrics_to_visualize = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        else:
            metrics_to_visualize = ['RMSE', 'MAE', 'R² Score']
        
        # Filter to only metrics that exist in the data
        metrics_to_visualize = [m for m in metrics_to_visualize if m in comparison_df.columns]
        
        if not metrics_to_visualize:
            st.warning("No common metrics found for visualization.")
            return
        
        # Let user select which metric to visualize
        selected_metric = st.selectbox("Select metric to visualize", metrics_to_visualize)
        
        if selected_metric:
            # Prepare data for bar chart
            plot_data = comparison_df[['Model', selected_metric]].sort_values(selected_metric, ascending=False)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Decide colors based on metric
            if selected_metric in ['RMSE', 'MAE']:  # Lower is better
                colors = ['#ff9999' if x > plot_data[selected_metric].min() else '#99ff99' for x in plot_data[selected_metric]]
            else:  # Higher is better
                colors = ['#99ff99' if x >= plot_data[selected_metric].max() * 0.95 else '#ff9999' for x in plot_data[selected_metric]]
            
            bars = ax.bar(plot_data['Model'], plot_data[selected_metric], color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            ax.set_xlabel('Model')
            ax.set_ylabel(selected_metric)
            ax.set_title(f'Model Comparison by {selected_metric}')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig) 