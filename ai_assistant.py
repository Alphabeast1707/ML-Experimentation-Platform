import streamlit as st
import pandas as pd
import numpy as np
import json

class AIAssistantTab:
    def render(self):
        st.markdown("<h2 class='sub-header'>AI Assistant</h2>", unsafe_allow_html=True)
        
        # Introduction
        st.markdown("""
        <div class="info-box">
            <h3>🤖 AI Assistant for ML Experimentation</h3>
            <p>Get intelligent recommendations to improve your models and preprocessing steps based on your data and current results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for different assistant capabilities
        assistant_tabs = st.tabs(["Model Recommendations", "Feature Engineering", "Hyperparameter Tuning", "Error Analysis"])
        
        # Model Recommendations
        with assistant_tabs[0]:
            self.model_recommendations()
        
        # Feature Engineering
        with assistant_tabs[1]:
            self.feature_engineering_recommendations()
        
        # Hyperparameter Tuning
        with assistant_tabs[2]:
            self.hyperparameter_tuning_recommendations()
        
        # Error Analysis
        with assistant_tabs[3]:
            self.error_analysis()
    
    def model_recommendations(self):
        st.subheader("Model Recommendations")
        
        # Check if data is loaded
        if st.session_state.dataset is None:
            st.warning("⚠️ No dataset loaded. Please load a dataset first.")
            return
        
        # Get dataset and problem type information
        dataset_info = st.session_state.dataset_info
        problem_type = st.session_state.problem_type
        
        # Display dataset overview
        if dataset_info:
            st.write(f"**Dataset:** {dataset_info.get('name', 'Unknown')}")
            st.write(f"**Shape:** {dataset_info.get('shape', (0, 0))}")
            st.write(f"**Problem Type:** {problem_type if problem_type else 'Not defined'}")
        
        # Recommendations based on dataset characteristics
        st.markdown("### 📊 Recommended Models")
        
        if not problem_type:
            st.info("Please define the problem type in the Data Preprocessing tab to get model recommendations.")
            return
        
        # Get dataset characteristics for recommendation logic
        data_size = len(st.session_state.dataset)
        num_features = len(st.session_state.feature_columns) if st.session_state.feature_columns else 0
        
        # Simple recommendation logic based on dataset characteristics
        if problem_type == "Classification":
            st.markdown("""
            <div class="metric-card">
                <h4>Classification Model Recommendations</h4>
            """, unsafe_allow_html=True)
            
            if data_size < 1000:
                st.markdown("""
                <p><strong>For Small Datasets:</strong></p>
                <ul>
                    <li>🥇 Random Forest: Generally performs well on small datasets with good accuracy and less overfitting</li>
                    <li>🥈 SVM with RBF kernel: Good for complex decision boundaries in smaller datasets</li>
                    <li>🥉 Logistic Regression: Simple, interpretable baseline with regularization to prevent overfitting</li>
                </ul>
                """, unsafe_allow_html=True)
            elif 1000 <= data_size < 10000:
                st.markdown("""
                <p><strong>For Medium Datasets:</strong></p>
                <ul>
                    <li>🥇 Random Forest: Still a strong performer with good balance of accuracy and training time</li>
                    <li>🥈 XGBoost: Can achieve high accuracy with proper tuning</li>
                    <li>🥉 Neural Network: Consider a simple 2-3 layer network with dropout</li>
                </ul>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <p><strong>For Large Datasets:</strong></p>
                <ul>
                    <li>🥇 Deep Neural Network: Performance scales well with data volume, try networks with multiple layers</li>
                    <li>🥈 Gradient Boosting (XGBoost): High accuracy and reasonable training time</li>
                    <li>🥉 Random Forest: Still good, but may have longer training times</li>
                </ul>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:  # Regression
            st.markdown("""
            <div class="metric-card">
                <h4>Regression Model Recommendations</h4>
            """, unsafe_allow_html=True)
            
            if data_size < 1000:
                st.markdown("""
                <p><strong>For Small Datasets:</strong></p>
                <ul>
                    <li>🥇 Ridge Regression: Linear model with L2 regularization to prevent overfitting</li>
                    <li>🥈 Random Forest Regressor: Good performance with minimal tuning</li>
                    <li>🥉 SVR with RBF kernel: Good for complex relationships in smaller datasets</li>
                </ul>
                """, unsafe_allow_html=True)
            elif 1000 <= data_size < 10000:
                st.markdown("""
                <p><strong>For Medium Datasets:</strong></p>
                <ul>
                    <li>🥇 Random Forest Regressor: Good balance of accuracy and training time</li>
                    <li>🥈 Gradient Boosting Regressor: Can achieve high accuracy with proper tuning</li>
                    <li>🥉 Neural Network: Consider a simple 2-3 layer network</li>
                </ul>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <p><strong>For Large Datasets:</strong></p>
                <ul>
                    <li>🥇 Deep Neural Network: Can model complex relationships, especially with many features</li>
                    <li>🥈 Gradient Boosting (XGBoost): Great balance of performance and training time</li>
                    <li>🥉 Random Forest Regressor: Reliable performance but may be slower to train</li>
                </ul>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional recommendations based on dataset characteristics
        if st.session_state.feature_columns:
            categorical_cols = st.session_state.dataset[st.session_state.feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = st.session_state.dataset[st.session_state.feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(categorical_cols) > len(numeric_cols) * 2:
                st.markdown("""
                <div class="info-box">
                    <p>⚠️ <strong>Note:</strong> Your dataset has many categorical features. Consider:</p>
                    <ul>
                        <li>Tree-based models like Random Forest handle categorical features well</li>
                        <li>For neural networks, use proper encoding techniques (embedding layers for high cardinality)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Ask for more detailed recommendations
        st.markdown("### 🔍 Need more specific recommendations?")
        specific_request = st.text_area("Describe your specific needs or challenges:", placeholder="E.g., 'My model is overfitting with high variance' or 'I need a model that's faster to train'")
        
        if specific_request and st.button("Get Personalized Recommendations"):
            self.generate_personalized_recommendations(specific_request, problem_type)
    
    def feature_engineering_recommendations(self):
        st.subheader("Feature Engineering Recommendations")
        
        # Check if data is loaded
        if st.session_state.dataset is None:
            st.warning("⚠️ No dataset loaded. Please load a dataset first.")
            return
        
        # Get dataset information
        df = st.session_state.dataset
        
        # Data quality recommendations
        st.markdown("### 📊 Data Quality Recommendations")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df) * 100).round(2)
        missing_cols = missing_data[missing_data > 0]
        
        if len(missing_cols) > 0:
            st.markdown("""
            <div class="metric-card">
                <h4>Missing Value Handling</h4>
                <p>Your dataset has columns with missing values:</p>
            """, unsafe_allow_html=True)
            
            # Display columns with missing values
            missing_summary = pd.DataFrame({
                'Column': missing_cols.index,
                'Missing Count': missing_cols.values,
                'Missing Percentage': missing_percentage[missing_cols.index].values
            }).sort_values('Missing Count', ascending=False)
            
            st.dataframe(missing_summary, use_container_width=True)
            
            # Missing value recommendations
            st.markdown("""
            <p><strong>Recommendations:</strong></p>
            <ul>
            """, unsafe_allow_html=True)
            
            for col, pct in zip(missing_summary['Column'], missing_summary['Missing Percentage']):
                if pct > 50:
                    st.markdown(f"<li>❌ Consider dropping the column '{col}' as it has {pct:.1f}% missing values</li>", unsafe_allow_html=True)
                elif pct > 30:
                    st.markdown(f"<li>⚠️ For '{col}' with {pct:.1f}% missing, consider advanced imputation methods like KNN or model-based imputation</li>", unsafe_allow_html=True)
                else:
                    if df[col].dtype in ['int64', 'float64']:
                        st.markdown(f"<li>✅ For numeric column '{col}', impute with median (more robust to outliers than mean)</li>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<li>✅ For categorical column '{col}', impute with mode or create a 'Missing' category</li>", unsafe_allow_html=True)
            
            st.markdown("""
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Missing Value Handling</h4>
                <p>✅ Great! Your dataset has no missing values.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature creation recommendations
        st.markdown("### 🛠️ Feature Creation Recommendations")
        
        # Numeric features
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.markdown("""
            <div class="metric-card">
                <h4>Numeric Feature Engineering</h4>
                <p><strong>Try these transformations:</strong></p>
                <ul>
                    <li>✨ <strong>Polynomial Features:</strong> Create interaction terms between features (x₁×x₂, x₁², x₂²)</li>
                    <li>✨ <strong>Binning/Discretization:</strong> Convert continuous variables into discrete bins</li>
                    <li>✨ <strong>Log Transformation:</strong> Apply to skewed features to make distributions more normal</li>
                    <li>✨ <strong>Normalization/Standardization:</strong> Scale features to similar ranges</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            st.markdown("""
            <div class="metric-card">
                <h4>Categorical Feature Engineering</h4>
                <p><strong>Encoding recommendations:</strong></p>
                <ul>
            """, unsafe_allow_html=True)
            
            for col in categorical_cols:
                cardinality = df[col].nunique()
                if cardinality > 10:
                    st.markdown(f"<li>🔍 '{col}' has high cardinality ({cardinality} categories). Consider target encoding or embedding.</li>", unsafe_allow_html=True)
                elif 5 < cardinality <= 10:
                    st.markdown(f"<li>🔍 '{col}' has moderate cardinality ({cardinality} categories). One-hot encoding is appropriate.</li>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<li>🔍 '{col}' has low cardinality ({cardinality} categories). One-hot encoding works well.</li>", unsafe_allow_html=True)
            
            st.markdown("""
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature selection recommendations
        st.markdown("### 🔍 Feature Selection Recommendations")
        
        if len(numeric_cols) > 10:
            st.markdown("""
            <div class="metric-card">
                <h4>Feature Selection Methods</h4>
                <p>Your dataset has many numeric features. Consider these feature selection methods:</p>
                <ul>
                    <li>✂️ <strong>Principal Component Analysis (PCA):</strong> Reduce dimensions while preserving variance</li>
                    <li>✂️ <strong>SelectKBest:</strong> Choose features based on statistical tests</li>
                    <li>✂️ <strong>Recursive Feature Elimination:</strong> Iteratively remove least important features</li>
                    <li>✂️ <strong>Feature Importance from Tree Models:</strong> Use Random Forest to rank features</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def hyperparameter_tuning_recommendations(self):
        st.subheader("Hyperparameter Tuning Recommendations")
        
        # Check if a model has been trained
        if st.session_state.model is None:
            st.warning("⚠️ No model has been trained yet. Please train a model first.")
            return
        
        # Get model information
        model_type = st.session_state.model_type
        model_category = st.session_state.selected_model_category
        problem_type = st.session_state.problem_type
        
        st.markdown(f"**Current Model:** {model_type}")
        st.markdown(f"**Model Category:** {model_category}")
        st.markdown(f"**Problem Type:** {problem_type}")
        
        # Display hyperparameter recommendations based on model type
        st.markdown("### 🎛️ Key Hyperparameters to Tune")
        
        hyperparameter_recommendations = {
            "Random Forest": {
                "params": [
                    {"name": "n_estimators", "importance": "High", "recommendation": "Start with 100-200, increase for better performance"},
                    {"name": "max_depth", "importance": "High", "recommendation": "Control tree depth to prevent overfitting (try 5-30)"},
                    {"name": "min_samples_split", "importance": "Medium", "recommendation": "Increasing prevents overfitting (try 2-20)"},
                    {"name": "min_samples_leaf", "importance": "Medium", "recommendation": "Higher values prevent complex trees (try 1-10)"},
                    {"name": "max_features", "importance": "Medium", "recommendation": "'sqrt' or 'log2' for classification, higher values for regression"}
                ],
                "search_strategy": "Random search is usually effective; prioritize n_estimators and max_depth"
            },
            "Logistic Regression": {
                "params": [
                    {"name": "C", "importance": "High", "recommendation": "Inverse of regularization strength (try 0.001 to 100 on log scale)"},
                    {"name": "penalty", "importance": "Medium", "recommendation": "'l1' for feature selection, 'l2' default, 'elasticnet' for combined"},
                    {"name": "solver", "importance": "Low", "recommendation": "'liblinear' for small datasets, 'saga' for large ones"},
                    {"name": "class_weight", "importance": "Medium", "recommendation": "'balanced' for imbalanced classes"}
                ],
                "search_strategy": "Grid search with focus on regularization (C) and penalty"
            },
            "SVM": {
                "params": [
                    {"name": "C", "importance": "High", "recommendation": "Higher values: less regularization (try 0.1-100 on log scale)"},
                    {"name": "kernel", "importance": "High", "recommendation": "'rbf' for non-linear data, 'linear' for high-dim data"},
                    {"name": "gamma", "importance": "High", "recommendation": "For 'rbf', controls decision boundary (try 0.001-10 on log scale)"}
                ],
                "search_strategy": "Grid search with C and gamma using log scales"
            },
            "Dense Neural Network": {
                "params": [
                    {"name": "learning_rate", "importance": "High", "recommendation": "Try 0.01, 0.001, 0.0001"},
                    {"name": "batch_size", "importance": "Medium", "recommendation": "Higher for efficiency, lower for learning (32-256)"},
                    {"name": "number_of_layers", "importance": "Medium", "recommendation": "2-5 hidden layers for most problems"},
                    {"name": "neurons_per_layer", "importance": "Medium", "recommendation": "Start with powers of 2 (64, 128, 256)"},
                    {"name": "dropout_rate", "importance": "High", "recommendation": "0.2-0.5 to prevent overfitting"},
                    {"name": "activation", "importance": "Medium", "recommendation": "'relu' for hidden layers, appropriate output activation for problem"}
                ],
                "search_strategy": "Start with learning rate and dropout, then network architecture"
            },
            "CNN": {
                "params": [
                    {"name": "learning_rate", "importance": "High", "recommendation": "Try 0.01, 0.001, 0.0001"},
                    {"name": "number_of_filters", "importance": "Medium", "recommendation": "Increase progressively in deeper layers (32→64→128)"},
                    {"name": "kernel_size", "importance": "Medium", "recommendation": "3x3 is standard, 5x5 for larger patterns"},
                    {"name": "pooling", "importance": "Low", "recommendation": "MaxPooling2D with 2x2 pool size typically works well"}
                ],
                "search_strategy": "Learning rate first, then filter architecture"
            },
            "RNN/LSTM": {
                "params": [
                    {"name": "learning_rate", "importance": "High", "recommendation": "Lower than usual (0.001, 0.0005)"},
                    {"name": "units", "importance": "Medium", "recommendation": "32-256 units per layer"},
                    {"name": "recurrent_dropout", "importance": "High", "recommendation": "0.1-0.3 to prevent overfitting"},
                    {"name": "number_of_layers", "importance": "Medium", "recommendation": "1-3 for most sequence problems"}
                ],
                "search_strategy": "Learning rate and units are most important"
            }
        }
        
        # Find the closest model type in our recommendations
        best_match = None
        for key in hyperparameter_recommendations:
            if key in model_type:
                best_match = key
                break
        
        if best_match is None:
            # Try to find a generic match based on model category
            if model_category == "Neural Networks":
                best_match = "Dense Neural Network"
            elif "Forest" in model_type:
                best_match = "Random Forest"
            elif "SVM" in model_type or "SVR" in model_type:
                best_match = "SVM"
        
        if best_match:
            recommendations = hyperparameter_recommendations[best_match]
            
            # Create expandable sections for each parameter
            for param in recommendations["params"]:
                importance_color = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟡"
                }
                
                st.markdown(f"#### {importance_color[param['importance']]} {param['name']}")
                st.markdown(f"**Importance:** {param['importance']}")
                st.markdown(f"**Recommendation:** {param['recommendation']}")
            
            st.markdown("### 🔍 Search Strategy")
            st.markdown(recommendations["search_strategy"])
        else:
            st.info("Detailed recommendations not available for this model type. Try generic tuning approaches like random search or Bayesian optimization.")
        
        # General tuning advice
        st.markdown("### 📈 General Tuning Advice")
        st.markdown("""
        <div class="info-box">
            <p><strong>Effective Tuning Process:</strong></p>
            <ol>
                <li>Start with a baseline model using default parameters</li>
                <li>Perform initial exploration with a wide parameter range</li>
                <li>Narrow down ranges based on initial results</li>
                <li>Use cross-validation to ensure robust performance</li>
                <li>Consider the bias-variance tradeoff (complexity vs. generalization)</li>
            </ol>
            <p><strong>Methods:</strong></p>
            <ul>
                <li><strong>Grid Search:</strong> Exhaustive but expensive for many parameters</li>
                <li><strong>Random Search:</strong> More efficient for high-dimensional spaces</li>
                <li><strong>Bayesian Optimization:</strong> Learns from previous evaluations, good for expensive models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def error_analysis(self):
        st.subheader("Error Analysis")
        
        # Check if a model has been trained and evaluated
        if st.session_state.experiment_results is None:
            st.warning("⚠️ No model has been evaluated yet. Please train and evaluate a model first.")
            return
        
        # Get experiment results
        results = st.session_state.experiment_results
        problem_type = results["problem_type"]
        
        st.markdown("### 🔍 Error Analysis")
        
        if problem_type == "Classification":
            self.classification_error_analysis(results)
        else:
            self.regression_error_analysis(results)
    
    def classification_error_analysis(self, results):
        # Extract data
        y_true = results["y_test"]
        y_pred = results["y_pred"]
        
        # Convert to arrays
        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()
        
        # Identify misclassifications
        misclassified_indices = np.where(y_true_flat != y_pred_flat)[0]
        
        if len(misclassified_indices) == 0:
            st.success("✅ Perfect classification! No errors to analyze.")
            return
        
        # Compute error rate
        error_rate = len(misclassified_indices) / len(y_true_flat) * 100
        
        st.markdown(f"**Error Rate:** {error_rate:.2f}% ({len(misclassified_indices)} out of {len(y_true_flat)} samples)")
        
        # Confusion analysis - Find the most common misclassifications
        unique_true = np.unique(y_true_flat)
        unique_pred = np.unique(y_pred_flat)
        
        error_counts = {}
        for true_class in unique_true:
            for pred_class in unique_pred:
                if true_class != pred_class:
                    count = np.sum((y_true_flat == true_class) & (y_pred_flat == pred_class))
                    if count > 0:
                        error_counts[(true_class, pred_class)] = count
        
        # Sort errors by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_errors:
            st.markdown("#### Most Common Misclassifications")
            
            for (true_class, pred_class), count in sorted_errors[:5]:  # Show top 5
                st.markdown(f"True class **{true_class}** predicted as **{pred_class}**: {count} instances ({count/len(y_true_flat)*100:.2f}%)")
            
            # Recommendations based on error patterns
            st.markdown("#### Recommendations for Improvement")
            
            # General recommendations based on common error patterns
            st.markdown("""
            <div class="metric-card">
                <h4>Strategies to Improve Classification:</h4>
                <ul>
            """, unsafe_allow_html=True)
            
            # Check for class imbalance
            class_counts = np.bincount(y_true_flat.astype(int))
            max_class = class_counts.max()
            min_class = class_counts.min()
            imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
            
            if imbalance_ratio > 5:
                st.markdown("""
                <li>⚠️ <strong>Class Imbalance Detected:</strong> Try these techniques:
                    <ul>
                        <li>Resampling: Oversample minority classes or undersample majority class</li>
                        <li>Use class weights in your model</li>
                        <li>Try SMOTE or other synthetic data generation techniques</li>
                    </ul>
                </li>
                """, unsafe_allow_html=True)
            
            # Feature engineering advice
            st.markdown("""
            <li>✨ <strong>Feature Engineering:</strong> Create new features that help distinguish between the most confused classes</li>
            <li>🔍 <strong>Model Complexity:</strong> If errors seem systematic, try a more complex model; if they seem random, your model might be overfitting</li>
            <li>🧪 <strong>Ensemble Methods:</strong> Combine multiple models to reduce errors, especially for the most problematic classes</li>
            <li>📊 <strong>Error Sampling:</strong> Collect more training data focused on the most confused classes</li>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def regression_error_analysis(self, results):
        # Extract data
        y_true = results["y_test"]
        y_pred = results["y_pred"]
        
        # Convert to arrays
        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()
        
        # Calculate errors
        errors = y_true_flat - y_pred_flat
        abs_errors = np.abs(errors)
        
        # Calculate key error metrics
        mean_error = np.mean(errors)
        mean_abs_error = np.mean(abs_errors)
        
        st.markdown(f"**Mean Error:** {mean_error:.4f}")
        st.markdown(f"**Mean Absolute Error:** {mean_abs_error:.4f}")
        
        # Check if there are features to analyze
        if hasattr(st.session_state, 'train_test_data') and 'X_test' in st.session_state.train_test_data:
            X_test = st.session_state.train_test_data['X_test']
            
            # Identify patterns in errors
            st.markdown("#### Error Patterns Analysis")
            
            # Find largest errors
            large_error_threshold = np.percentile(abs_errors, 90)  # Top 10% of errors
            large_error_indices = np.where(abs_errors > large_error_threshold)[0]
            
            if len(large_error_indices) > 0:
                st.markdown(f"Analyzing {len(large_error_indices)} samples with the largest errors (above {large_error_threshold:.4f})")
                
                # Check where the model tends to overpredict or underpredict
                overprediction_count = np.sum(errors < 0)
                underprediction_count = np.sum(errors > 0)
                
                if overprediction_count > len(errors) * 0.6:
                    st.markdown("⚠️ **Pattern detected:** Model tends to **overpredict** values (predictions higher than actual)")
                elif underprediction_count > len(errors) * 0.6:
                    st.markdown("⚠️ **Pattern detected:** Model tends to **underpredict** values (predictions lower than actual)")
                
                # Recommendations based on error analysis
                st.markdown("#### Recommendations for Improvement")
                
                st.markdown("""
                <div class="metric-card">
                    <h4>Strategies to Improve Regression:</h4>
                    <ul>
                """, unsafe_allow_html=True)
                
                # Nonlinearity check - simple heuristic based on error distribution
                if np.abs(np.percentile(errors, 75) + np.percentile(errors, 25)) > 0.1 * mean_abs_error:
                    st.markdown("""
                    <li>🔍 <strong>Non-linearity detected:</strong> Consider:
                        <ul>
                            <li>Adding polynomial features or interaction terms</li>
                            <li>Using more flexible models like Random Forest or Neural Networks</li>
                            <li>Applying non-linear transformations to key features</li>
                        </ul>
                    </li>
                    """, unsafe_allow_html=True)
                
                # Check for potential outliers
                outlier_threshold = np.percentile(abs_errors, 98)  # Top 2% as outliers
                outlier_count = np.sum(abs_errors > outlier_threshold)
                
                if outlier_count > 0:
                    st.markdown(f"""
                    <li>⚠️ <strong>Potential outliers detected:</strong> {outlier_count} samples have extremely large errors. Consider:
                        <ul>
                            <li>Examining these samples for data quality issues</li>
                            <li>Using robust regression methods</li>
                            <li>Removing extreme outliers if they represent errors</li>
                        </ul>
                    </li>
                    """, unsafe_allow_html=True)
                
                # General recommendations
                st.markdown("""
                <li>✨ <strong>Feature Engineering:</strong> Create new features that might better capture the relationship</li>
                <li>🧮 <strong>Try different loss functions:</strong> If large errors are concerning, try Huber loss or quantile regression</li>
                <li>🔄 <strong>Data transformations:</strong> If target distribution is skewed, try log or other transformations</li>
                <li>🧪 <strong>Ensemble methods:</strong> Combine multiple models to reduce error variance</li>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    def generate_personalized_recommendations(self, query, problem_type):
        """Generate personalized AI recommendations based on user query"""
        st.markdown("### 💡 Personalized Recommendations")
        
        # Common issues and solutions based on keywords in the query
        keywords = {
            "overfitting": [
                "Increase regularization strength (L1/L2)",
                "Add dropout layers for neural networks",
                "Reduce model complexity (fewer layers, fewer trees)",
                "Collect more training data",
                "Use early stopping during training"
            ],
            "underfitting": [
                "Increase model complexity (more layers, more trees)",
                "Reduce regularization strength",
                "Create more engineered features",
                "Try more powerful model architectures",
                "Train for more epochs (neural networks)"
            ],
            "slow": [
                "Try more efficient model implementations",
                "Reduce dataset size with sampling if possible",
                "Use fewer features through feature selection",
                "Try simpler model architectures",
                "Consider GPU acceleration for neural networks"
            ],
            "imbalanced": [
                "Use class weighting in your model",
                "Try resampling techniques (SMOTE, undersampling)",
                "Use ensemble methods like balanced bagging",
                "Change your evaluation metric (F1, AUC instead of accuracy)",
                "Collect more data for minority classes"
            ],
            "accuracy": [
                "Ensemble multiple model types",
                "Perform more extensive hyperparameter tuning",
                "Try more advanced architectures",
                "Focus on better feature engineering",
                "Consider domain-specific preprocessing steps"
            ]
        }
        
        # Find relevant keywords in the query
        relevant_solutions = []
        for keyword, solutions in keywords.items():
            if keyword.lower() in query.lower():
                relevant_solutions.extend(solutions)
        
        # If no specific keywords match, provide general recommendations
        if not relevant_solutions:
            if problem_type == "Classification":
                relevant_solutions = [
                    "Try ensemble methods like Random Forest or XGBoost",
                    "Experiment with different neural network architectures",
                    "Perform cross-validation to ensure model generalization",
                    "Focus on feature engineering to create more predictive inputs",
                    "Check for class imbalance and address if present"
                ]
            else:  # Regression
                relevant_solutions = [
                    "Try ensemble methods like Gradient Boosting or Random Forest",
                    "Add polynomial features to capture non-linear relationships",
                    "Check the distribution of your target variable",
                    "Look for outliers that might be affecting model performance",
                    "Consider different error metrics based on your specific needs"
                ]
        
        # Display personalized recommendations
        st.markdown("""
        <div class="metric-card">
            <h4>Based on your request, here are personalized recommendations:</h4>
            <ul>
        """, unsafe_allow_html=True)
        
        for solution in list(set(relevant_solutions))[:5]:  # Remove duplicates and show top 5
            st.markdown(f"<li>💡 {solution}</li>", unsafe_allow_html=True)
        
        st.markdown("""
            </ul>
        </div>
        """, unsafe_allow_html=True) 