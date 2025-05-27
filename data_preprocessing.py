import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
import pickle
import io
import base64
from io import StringIO
from sklearn.datasets import fetch_california_housing, load_iris, load_wine, load_breast_cancer, load_diabetes

class DataPreprocessingTab:
    def render(self):
        st.markdown("<h2 class='sub-header'>Data Preprocessing</h2>", unsafe_allow_html=True)
        
        # Data Source Selection
        st.markdown("### 📊 Data Source")
        data_source = st.radio(
            "Select Data Source",
            ["Upload Dataset", "Built-in Dataset", "Load Saved Dataset"]
        )
        
        # Handle different data sources
        if data_source == "Upload Dataset":
            self.handle_uploaded_dataset()
        elif data_source == "Built-in Dataset":
            self.handle_builtin_dataset()
        elif data_source == "Load Saved Dataset":
            self.handle_saved_dataset()
        
        # Proceed with data preprocessing if dataset is loaded
        if st.session_state.dataset is not None:
            self.data_preprocessing_pipeline()
    
    def handle_uploaded_dataset(self):
        st.markdown("### 📤 Upload Your Dataset")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Check file extension to determine how to read it
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.dataset = df
                st.session_state.dataset_info = {
                    "name": uploaded_file.name,
                    "description": "User uploaded dataset",
                    "shape": df.shape
                }
                st.success(f"✅ Dataset '{uploaded_file.name}' loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns!")
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
    
    def handle_builtin_dataset(self):
        st.markdown("### 📊 Built-in Datasets")
        
        dataset_options = {
            "California Housing": "Regression dataset with housing prices in California districts",
            "Iris": "Classification dataset with iris flower measurements",
            "Wine": "Classification dataset with chemical analysis of wines",
            "Breast Cancer": "Binary classification dataset for breast cancer diagnosis",
            "Diabetes": "Regression dataset for diabetes progression"
        }
        
        # Dataset selection with descriptions
        selected_dataset = st.selectbox(
            "Select Dataset",
            list(dataset_options.keys())
        )
        
        st.info(dataset_options[selected_dataset])
        
        # Load button
        if st.button("Load Dataset"):
            with st.spinner(f"Loading {selected_dataset} dataset..."):
                if selected_dataset == "California Housing":
                    data = fetch_california_housing()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                    description = "A dataset of housing prices in California"
                
                elif selected_dataset == "Iris":
                    data = load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                    description = "A dataset of iris flower measurements"
                
                elif selected_dataset == "Wine":
                    data = load_wine()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                    description = "A dataset of wine chemical properties"
                
                elif selected_dataset == "Breast Cancer":
                    data = load_breast_cancer()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                    description = "A dataset for breast cancer diagnosis"
                
                elif selected_dataset == "Diabetes":
                    data = load_diabetes()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                    description = "A dataset for diabetes progression"
                
                st.session_state.dataset = df
                st.session_state.dataset_info = {
                    "name": selected_dataset,
                    "description": description,
                    "shape": df.shape,
                    "feature_names": list(df.columns[:-1]),
                    "target_name": "target"
                }
                st.success(f"✅ {selected_dataset} dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns!")
    
    def handle_saved_dataset(self):
        st.markdown("### 💾 Load Saved Dataset")
        
        # Check for saved datasets in the data directory
        data_dir = os.path.join("data")
        os.makedirs(data_dir, exist_ok=True)
        
        saved_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        
        if not saved_files:
            st.info("No saved datasets found. Process and save a dataset first.")
            return
        
        selected_file = st.selectbox("Select a saved dataset", saved_files)
        
        if st.button("Load Selected Dataset"):
            try:
                with open(os.path.join(data_dir, selected_file), 'rb') as f:
                    saved_data = pickle.load(f)
                
                st.session_state.dataset = saved_data.get('processed_data')
                st.session_state.train_test_data = saved_data.get('train_test_data')
                st.session_state.feature_columns = saved_data.get('feature_columns')
                st.session_state.target_column = saved_data.get('target_column')
                st.session_state.problem_type = saved_data.get('problem_type')
                st.session_state.dataset_info = saved_data.get('dataset_info')
                
                st.success(f"✅ Dataset '{selected_file}' loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
    
    def data_preprocessing_pipeline(self):
        df = st.session_state.dataset
        
        # Basic data info
        st.markdown("### 📋 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        
        # Data preview with tabs
        data_tabs = st.tabs(["Preview", "Summary", "Visualizations"])
        
        with data_tabs[0]:
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download original data as CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Original Data",
                csv,
                "original_data.csv",
                "text/csv",
                key='download-original-csv'
            )
        
        with data_tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Types")
                st.dataframe(pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Unique Values': [df[col].nunique() for col in df.columns],
                    'Missing Values': df.isna().sum(),
                    '% Missing': (df.isna().sum() / len(df) * 100).round(2)
                }), use_container_width=True)
            
            with col2:
                st.subheader("Statistical Summary")
                st.dataframe(df.describe().transpose(), use_container_width=True)
        
        with data_tabs[2]:
            st.subheader("Data Visualizations")
            
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if numeric_cols:
                    selected_num_col = st.selectbox("Select Numeric Column for Distribution", numeric_cols)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[selected_num_col].dropna(), kde=True, ax=ax)
                    ax.set_title(f'Distribution of {selected_num_col}')
                    st.pyplot(fig)
            
            with vis_col2:
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Select X Column", numeric_cols, index=0)
                    y_col = st.selectbox("Select Y Column", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                    ax.set_title(f'{x_col} vs {y_col}')
                    st.pyplot(fig)
            
            # Correlation Matrix for numeric columns
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
                st.pyplot(fig)
        
        # Data Preprocessing Steps
        st.markdown("### 🔄 Data Preprocessing")
        
        preprocessing_tabs = st.tabs([
            "Problem Type", 
            "Target & Features", 
            "Missing Values", 
            "Encoding", 
            "Scaling", 
            "Feature Selection/Extraction", 
            "Train-Test Split"
        ])
        
        # 1. Problem Type
        with preprocessing_tabs[0]:
            st.subheader("Step 1: Define Problem Type")
            problem_type = st.radio(
                "Select Problem Type",
                ["Classification", "Regression"],
                index=0 if st.session_state.problem_type == "Classification" else 1
            )
            st.session_state.problem_type = problem_type
            
            if st.button("Confirm Problem Type"):
                st.success(f"Problem type set to {problem_type}")
        
        # 2. Target & Features Selection
        with preprocessing_tabs[1]:
            st.subheader("Step 2: Select Target & Features")
            
            all_columns = df.columns.tolist()
            
            # Target selection
            target_column = st.selectbox(
                "Select Target Column",
                all_columns,
                index=all_columns.index(st.session_state.target_column) if st.session_state.target_column in all_columns else 0
            )
            
            # Feature selection
            feature_cols = [col for col in all_columns if col != target_column]
            default_selected = st.session_state.feature_columns if st.session_state.feature_columns else feature_cols
            selected_features = st.multiselect(
                "Select Feature Columns",
                feature_cols,
                default=default_selected
            )
            
            if st.button("Confirm Target & Features"):
                if not selected_features:
                    st.error("❌ You must select at least one feature column!")
                else:
                    st.session_state.target_column = target_column
                    st.session_state.feature_columns = selected_features
                    st.success(f"✅ Target: {target_column}, Features: {len(selected_features)} columns selected")
        
        # 3. Missing Values Handling
        with preprocessing_tabs[2]:
            st.subheader("Step 3: Handle Missing Values")
            
            if st.session_state.feature_columns and st.session_state.target_column:
                X = df[st.session_state.feature_columns]
                
                missing_data = X.isnull().sum().reset_index()
                missing_data.columns = ['Column', 'Missing Count']
                missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(X) * 100).round(2)
                missing_data = missing_data.sort_values('Missing Count', ascending=False)
                
                st.dataframe(missing_data, use_container_width=True)
                
                if missing_data['Missing Count'].sum() > 0:
                    st.subheader("Missing Values Strategy")
                    
                    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Numeric Columns Strategy")
                        numeric_strategy = st.selectbox(
                            "Select strategy for numeric columns",
                            ["mean", "median", "most_frequent", "constant", "knn"],
                            key="numeric_strategy"
                        )
                        numeric_fill_value = st.number_input(
                            "Fill value (for constant strategy)",
                            value=0,
                            key="numeric_fill_value"
                        )
                    
                    with col2:
                        st.write("Categorical Columns Strategy")
                        categorical_strategy = st.selectbox(
                            "Select strategy for categorical columns",
                            ["most_frequent", "constant"],
                            key="categorical_strategy"
                        )
                        categorical_fill_value = st.text_input(
                            "Fill value (for constant strategy)",
                            value="missing",
                            key="categorical_fill_value"
                        )
                    
                    if st.button("Apply Missing Values Handling"):
                        with st.spinner("Handling missing values..."):
                            X_copy = X.copy()
                            
                            # Handle numeric columns
                            if numeric_cols and set(numeric_cols).intersection(set(X.columns[X.isna().any()])):
                                if numeric_strategy == "knn":
                                    imputer = KNNImputer(n_neighbors=5)
                                    X_numeric = X_copy[numeric_cols]
                                    X_copy[numeric_cols] = imputer.fit_transform(X_numeric)
                                else:
                                    imputer = SimpleImputer(
                                        strategy=numeric_strategy,
                                        fill_value=numeric_fill_value if numeric_strategy == "constant" else None
                                    )
                                    X_copy[numeric_cols] = imputer.fit_transform(X_copy[numeric_cols])
                            
                            # Handle categorical columns
                            if categorical_cols and set(categorical_cols).intersection(set(X.columns[X.isna().any()])):
                                imputer = SimpleImputer(
                                    strategy=categorical_strategy,
                                    fill_value=categorical_fill_value if categorical_strategy == "constant" else None
                                )
                                
                                for col in categorical_cols:
                                    if X_copy[col].isna().any():
                                        X_copy[col] = imputer.fit_transform(X_copy[col].values.reshape(-1, 1)).ravel()
                            
                            # Update the dataframe with imputed values
                            for col in X_copy.columns:
                                df[col] = X_copy[col]
                            
                            # Also handle missing values in target column if any
                            if df[st.session_state.target_column].isna().any():
                                if df[st.session_state.target_column].dtype in ['int64', 'float64']:
                                    strategy = "mean" if st.session_state.problem_type == "Regression" else "most_frequent"
                                    imputer = SimpleImputer(strategy=strategy)
                                    df[st.session_state.target_column] = imputer.fit_transform(
                                        df[st.session_state.target_column].values.reshape(-1, 1)
                                    ).ravel()
                                else:
                                    imputer = SimpleImputer(strategy="most_frequent")
                                    df[st.session_state.target_column] = imputer.fit_transform(
                                        df[st.session_state.target_column].values.reshape(-1, 1)
                                    ).ravel()
                            
                            st.session_state.dataset = df
                            st.success("✅ Missing values handled successfully!")
                            
                            # Show updated missing values
                            X_updated = df[st.session_state.feature_columns]
                            missing_after = X_updated.isnull().sum().sum()
                            st.metric("Remaining Missing Values", missing_after, delta=missing_after - missing_data['Missing Count'].sum())
                else:
                    st.success("✅ No missing values found in the selected features!")
            else:
                st.warning("⚠️ Please select target and feature columns first!")
        
        # 4. Encoding
        with preprocessing_tabs[3]:
            st.subheader("Step 4: Categorical Encoding")
            
            if st.session_state.feature_columns:
                X = df[st.session_state.feature_columns]
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if categorical_cols:
                    st.write(f"Found {len(categorical_cols)} categorical columns:")
                    st.dataframe(pd.DataFrame({
                        'Column': categorical_cols,
                        'Unique Values': [X[col].nunique() for col in categorical_cols],
                        'Sample Values': [", ".join(X[col].dropna().sample(min(3, X[col].nunique())).astype(str).tolist()) for col in categorical_cols]
                    }), use_container_width=True)
                    
                    encoding_methods = {
                        "One-Hot Encoding": "Converts categorical columns to binary columns (recommended for nominal data)",
                        "Label Encoding": "Encodes categories as integers (recommended for ordinal data)",
                        "None": "Keep categorical columns as they are"
                    }
                    
                    encoding_method = st.radio(
                        "Select Encoding Method",
                        list(encoding_methods.keys()),
                        key="encoding_method"
                    )
                    
                    st.info(encoding_methods[encoding_method])
                    
                    if encoding_method != "None":
                        # Select columns to encode
                        cols_to_encode = st.multiselect(
                            "Select columns to encode",
                            categorical_cols,
                            default=categorical_cols
                        )
                        
                        if cols_to_encode and st.button("Apply Encoding"):
                            with st.spinner("Applying encoding..."):
                                df_encoded = df.copy()
                                
                                if encoding_method == "One-Hot Encoding":
                                    # Apply one-hot encoding
                                    for col in cols_to_encode:
                                        one_hot = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                                        df_encoded = pd.concat([df_encoded.drop(col, axis=1), one_hot], axis=1)
                                    
                                    # Update feature columns
                                    old_features = st.session_state.feature_columns.copy()
                                    new_features = [col for col in df_encoded.columns if col != st.session_state.target_column]
                                    st.session_state.feature_columns = new_features
                                
                                elif encoding_method == "Label Encoding":
                                    # Apply label encoding
                                    for col in cols_to_encode:
                                        le = LabelEncoder()
                                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                                
                                st.session_state.dataset = df_encoded
                                st.success(f"✅ Applied {encoding_method} to {len(cols_to_encode)} columns!")
                else:
                    st.info("No categorical columns found in selected features.")
            else:
                st.warning("⚠️ Please select feature columns first!")
        
        # 5. Scaling
        with preprocessing_tabs[4]:
            st.subheader("Step 5: Feature Scaling")
            
            if st.session_state.feature_columns:
                X = df[st.session_state.feature_columns]
                numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if numeric_cols:
                    scaling_methods = {
                        "StandardScaler": "Standardizes features by removing the mean and scaling to unit variance (z-score normalization)",
                        "MinMaxScaler": "Scales features to a given range, default [0,1]",
                        "RobustScaler": "Scales features using statistics that are robust to outliers",
                        "None": "No scaling applied"
                    }
                    
                    scaling_method = st.radio(
                        "Select Scaling Method",
                        list(scaling_methods.keys()),
                        key="scaling_method"
                    )
                    
                    st.info(scaling_methods[scaling_method])
                    
                    if scaling_method != "None":
                        # Select columns to scale
                        cols_to_scale = st.multiselect(
                            "Select columns to scale",
                            numeric_cols,
                            default=numeric_cols
                        )
                        
                        if cols_to_scale and st.button("Apply Scaling"):
                            with st.spinner("Applying scaling..."):
                                df_scaled = df.copy()
                                
                                if scaling_method == "StandardScaler":
                                    scaler = StandardScaler()
                                elif scaling_method == "MinMaxScaler":
                                    scaler = MinMaxScaler()
                                elif scaling_method == "RobustScaler":
                                    scaler = RobustScaler()
                                
                                df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
                                st.session_state.dataset = df_scaled
                                st.success(f"✅ Applied {scaling_method} to {len(cols_to_scale)} columns!")
                else:
                    st.info("No numeric columns found in selected features.")
            else:
                st.warning("⚠️ Please select feature columns first!")
        
        # 6. Feature Selection/Extraction
        with preprocessing_tabs[5]:
            st.subheader("Step 6: Feature Selection/Extraction")
            
            if st.session_state.feature_columns and st.session_state.target_column:
                X = df[st.session_state.feature_columns]
                y = df[st.session_state.target_column]
                
                feature_engineering_methods = [
                    "Select K Best Features",
                    "Principal Component Analysis (PCA)",
                    "None"
                ]
                
                method = st.radio(
                    "Select Feature Engineering Method",
                    feature_engineering_methods,
                    key="feature_engineering_method"
                )
                
                if method == "Select K Best Features":
                    st.info("Select the top K features based on statistical tests")
                    
                    k = st.slider(
                        "Number of features to select (K)",
                        min_value=1,
                        max_value=len(st.session_state.feature_columns),
                        value=min(5, len(st.session_state.feature_columns))
                    )
                    
                    score_funcs = {
                        "Classification": {
                            "ANOVA F-value": f_classif,
                            "Mutual Information": mutual_info_classif
                        },
                        "Regression": {
                            "F-value": f_regression,
                            "Mutual Information": mutual_info_regression
                        }
                    }
                    
                    score_func = st.selectbox(
                        "Score Function",
                        list(score_funcs[st.session_state.problem_type].keys())
                    )
                    
                    if st.button("Apply Feature Selection"):
                        with st.spinner("Selecting top features..."):
                            selector = SelectKBest(
                                score_funcs[st.session_state.problem_type][score_func],
                                k=k
                            )
                            
                            # Only use numeric columns for feature selection
                            X_numeric = X.select_dtypes(include=['int64', 'float64'])
                            
                            if X_numeric.shape[1] == 0:
                                st.error("❌ No numeric features available for selection!")
                            else:
                                X_new = selector.fit_transform(X_numeric, y)
                                
                                # Get selected feature names
                                selected_indices = selector.get_support(indices=True)
                                selected_features = X_numeric.columns[selected_indices].tolist()
                                
                                # Show feature scores
                                scores = pd.DataFrame({
                                    'Feature': X_numeric.columns,
                                    'Score': selector.scores_
                                }).sort_values('Score', ascending=False)
                                
                                st.dataframe(scores, use_container_width=True)
                                
                                # Update features list to include only selected numeric features and any non-numeric features
                                non_numeric_features = [col for col in st.session_state.feature_columns 
                                                        if col not in X_numeric.columns]
                                
                                st.session_state.feature_columns = selected_features + non_numeric_features
                                st.success(f"✅ Selected top {k} features: {', '.join(selected_features)}")
                
                elif method == "Principal Component Analysis (PCA)":
                    st.info("Reduce dimensionality while preserving variance")
                    
                    # Check if we have numeric features
                    X_numeric = X.select_dtypes(include=['int64', 'float64'])
                    
                    if X_numeric.shape[1] < 2:
                        st.error("❌ PCA requires at least 2 numeric features!")
                    else:
                        n_components = st.slider(
                            "Number of components",
                            min_value=1,
                            max_value=min(X_numeric.shape[1], 10),
                            value=min(2, X_numeric.shape[1])
                        )
                        
                        if st.button("Apply PCA"):
                            with st.spinner("Applying PCA..."):
                                pca = PCA(n_components=n_components)
                                X_pca = pca.fit_transform(X_numeric)
                                
                                # Create new DataFrame with PCA components
                                pca_cols = [f"PC{i+1}" for i in range(n_components)]
                                pca_df = pd.DataFrame(X_pca, columns=pca_cols)
                                
                                # Keep non-numeric columns
                                non_numeric_cols = [col for col in X.columns if col not in X_numeric.columns]
                                for col in non_numeric_cols:
                                    pca_df[col] = X[col].values
                                
                                # Add target column
                                pca_df[st.session_state.target_column] = y.values
                                
                                # Explained variance
                                explained_var = pca.explained_variance_ratio_
                                total_var = explained_var.sum() * 100
                                
                                # Component loadings
                                loadings = pd.DataFrame(
                                    pca.components_.T,
                                    columns=pca_cols,
                                    index=X_numeric.columns
                                )
                                
                                st.write("Explained Variance by Component:")
                                for i, var in enumerate(explained_var):
                                    st.write(f"PC{i+1}: {var*100:.2f}%")
                                st.write(f"Total Explained Variance: {total_var:.2f}%")
                                
                                st.write("Component Loadings:")
                                st.dataframe(loadings, use_container_width=True)
                                
                                # Update dataset and feature columns
                                st.session_state.dataset = pca_df
                                st.session_state.feature_columns = pca_cols + non_numeric_cols
                                st.success(f"✅ Applied PCA and reduced to {n_components} components")
            else:
                st.warning("⚠️ Please select target and feature columns first!")
        
        # 7. Train-Test Split
        with preprocessing_tabs[6]:
            st.subheader("Step 7: Train-Test Split")
            
            if st.session_state.feature_columns and st.session_state.target_column:
                X = df[st.session_state.feature_columns]
                y = df[st.session_state.target_column]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    test_size = st.slider(
                        "Test Size (%)",
                        min_value=10,
                        max_value=50,
                        value=20
                    ) / 100
                
                with col2:
                    random_state = st.number_input(
                        "Random State",
                        min_value=0,
                        max_value=100,
                        value=42
                    )
                
                stratify = False
                if st.session_state.problem_type == "Classification":
                    stratify = st.checkbox("Stratify by Target (Maintain class distribution)")
                
                if st.button("Create Train-Test Split"):
                    with st.spinner("Creating train-test split..."):
                        try:
                            stratify_param = y if stratify else None
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, 
                                test_size=test_size,
                                random_state=random_state,
                                stratify=stratify_param
                            )
                            
                            train_test_data = {
                                'X_train': X_train,
                                'X_test': X_test,
                                'y_train': y_train,
                                'y_test': y_test
                            }
                            
                            st.session_state.train_test_data = train_test_data
                            st.session_state.processed_data = df
                            
                            st.success(f"✅ Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
                            
                            # Show data distributions
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("Training set class distribution:")
                                if st.session_state.problem_type == "Classification":
                                    train_dist = pd.Series(y_train).value_counts(normalize=True) * 100
                                    st.dataframe(pd.DataFrame({
                                        'Class': train_dist.index,
                                        '% in Train': train_dist.values.round(2)
                                    }))
                                else:
                                    fig, ax = plt.subplots()
                                    ax.hist(y_train, bins=20)
                                    ax.set_title("Target Distribution (Train)")
                                    st.pyplot(fig)
                            
                            with col2:
                                st.write("Test set class distribution:")
                                if st.session_state.problem_type == "Classification":
                                    test_dist = pd.Series(y_test).value_counts(normalize=True) * 100
                                    st.dataframe(pd.DataFrame({
                                        'Class': test_dist.index,
                                        '% in Test': test_dist.values.round(2)
                                    }))
                                else:
                                    fig, ax = plt.subplots()
                                    ax.hist(y_test, bins=20)
                                    ax.set_title("Target Distribution (Test)")
                                    st.pyplot(fig)
                        except Exception as e:
                            st.error(f"❌ Error creating train-test split: {str(e)}")
            else:
                st.warning("⚠️ Please select target and feature columns first!")
        
        # Save Processed Data
        st.markdown("### 💾 Save Processed Data")
        
        if st.session_state.processed_data is not None and st.session_state.train_test_data is not None:
            save_name = st.text_input("Enter name for the processed dataset", "processed_dataset")
            
            if st.button("Save Processed Data"):
                try:
                    # Create a directory for data if it doesn't exist
                    data_dir = os.path.join("data")
                    os.makedirs(data_dir, exist_ok=True)
                    
                    # Save processed data
                    if not save_name.endswith('.pkl'):
                        save_name += '.pkl'
                    
                    data_path = os.path.join(data_dir, save_name)
                    with open(data_path, 'wb') as f:
                        pickle.dump({
                            'processed_data': st.session_state.processed_data,
                            'train_test_data': st.session_state.train_test_data,
                            'feature_columns': st.session_state.feature_columns,
                            'target_column': st.session_state.target_column,
                            'problem_type': st.session_state.problem_type,
                            'dataset_info': st.session_state.dataset_info
                        }, f)
                    
                    st.success(f"✅ Data saved successfully to {data_path}!")
                except Exception as e:
                    st.error(f"❌ Error saving data: {str(e)}")
        else:
            st.info("Complete the preprocessing steps to save processed data.")

def get_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href 