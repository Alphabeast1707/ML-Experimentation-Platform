"""
Data Processor — Handles data loading, EDA, transformations
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class DataProcessor:
    
    def get_sample_datasets(self):
        """Return list of available sample datasets"""
        return [
            {
                "name": "iris",
                "title": "🌸 Iris Flowers",
                "description": "Classic 3-class flower classification dataset. 150 samples, 4 features.",
                "rows": 150,
                "features": 4,
                "task": "Classification",
                "difficulty": "Beginner"
            },
            {
                "name": "wine",
                "title": "🍷 Wine Quality",
                "description": "Classify wines into 3 classes using chemical analysis. 178 samples, 13 features.",
                "rows": 178,
                "features": 13,
                "task": "Classification",
                "difficulty": "Beginner"
            },
            {
                "name": "breast_cancer",
                "title": "🏥 Breast Cancer",
                "description": "Binary classification of tumors as malignant or benign. 569 samples, 30 features.",
                "rows": 569,
                "features": 30,
                "task": "Classification",
                "difficulty": "Intermediate"
            },
            {
                "name": "diabetes",
                "title": "💉 Diabetes Progression",
                "description": "Predict diabetes disease progression. 442 samples, 10 features.",
                "rows": 442,
                "features": 10,
                "task": "Regression",
                "difficulty": "Intermediate"
            },
            {
                "name": "titanic",
                "title": "🚢 Titanic Survival",
                "description": "Predict passenger survival on the Titanic. 891 samples, 11 features.",
                "rows": 891,
                "features": 11,
                "task": "Classification",
                "difficulty": "Beginner"
            }
        ]
    
    def load_sample_dataset(self, name):
        """Load a sample dataset by name"""
        if name == "iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
            return df, "Iris flower classification dataset with 3 species"
        
        elif name == "wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            return df, "Wine recognition dataset with 3 classes"
        
        elif name == "breast_cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            df["target"] = df["target"].map({0: "malignant", 1: "benign"})
            return df, "Breast cancer diagnosis binary classification"
        
        elif name == "diabetes":
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            return df, "Diabetes disease progression prediction"
        
        elif name == "titanic":
            # Generate a realistic Titanic dataset
            np.random.seed(42)
            n = 891
            df = pd.DataFrame({
                "PassengerId": range(1, n + 1),
                "Pclass": np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
                "Sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
                "Age": np.random.normal(30, 14, n).clip(0.5, 80).round(1),
                "SibSp": np.random.choice([0, 1, 2, 3, 4], n, p=[0.60, 0.23, 0.10, 0.05, 0.02]),
                "Parch": np.random.choice([0, 1, 2, 3], n, p=[0.68, 0.17, 0.10, 0.05]),
                "Fare": np.random.exponential(33, n).round(2),
                "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09])
            })
            # Survival logic (somewhat realistic)
            survival_prob = 0.3 + 0.2 * (df["Sex"] == "female").astype(float) + 0.1 * (df["Pclass"] == 1).astype(float) - 0.05 * (df["Age"] > 50).astype(float)
            df["Survived"] = (np.random.random(n) < survival_prob).astype(int)
            return df, "Titanic passenger survival prediction"
        
        else:
            raise ValueError(f"Unknown sample dataset: {name}")
    
    def get_quick_stats(self, df):
        """Get quick summary statistics for a dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        missing_total = int(df.isnull().sum().sum())
        missing_pct = round(missing_total / (len(df) * len(df.columns)) * 100, 2) if len(df) > 0 else 0
        
        # Data quality score (0-100)
        quality_score = 100
        if missing_pct > 0:
            quality_score -= min(missing_pct * 2, 30)
        
        # Check for duplicates
        dup_pct = round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0
        if dup_pct > 5:
            quality_score -= min(dup_pct, 20)
        
        # Check for constant columns
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if constant_cols:
            quality_score -= len(constant_cols) * 5
        
        quality_score = max(0, round(quality_score))
        
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "missing_values": missing_total,
            "missing_percentage": missing_pct,
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": dup_pct,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "quality_score": quality_score,
            "column_names": list(df.columns)
        }
    
    def get_column_info(self, df):
        """Get detailed column information"""
        columns = []
        for col in df.columns:
            info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
                "unique": int(df[col].nunique()),
                "unique_pct": round(df[col].nunique() / len(df) * 100, 2) if len(df) > 0 else 0
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                info["is_numeric"] = True
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    info["mean"] = self._safe_float(non_null.mean())
                    info["std"] = self._safe_float(non_null.std())
                    info["min"] = self._safe_float(non_null.min())
                    info["max"] = self._safe_float(non_null.max())
                    info["median"] = self._safe_float(non_null.median())
                    info["skewness"] = self._safe_float(non_null.skew())
                else:
                    info["mean"] = info["std"] = info["min"] = info["max"] = info["median"] = info["skewness"] = None
            else:
                info["is_numeric"] = False
                try:
                    info["top_values"] = df[col].value_counts().head(5).to_dict()
                    # Convert keys to strings for JSON serialization
                    info["top_values"] = {str(k): int(v) for k, v in info["top_values"].items()}
                except Exception:
                    info["top_values"] = {}
            
            columns.append(info)
        return columns
    
    @staticmethod
    def _safe_float(val, default=0.0):
        """Safely convert a value to a JSON-serializable float"""
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return default
            return round(f, 4)
        except (TypeError, ValueError):
            return default

    def full_eda(self, df):
        """Perform comprehensive Exploratory Data Analysis"""
        result = {
            "overview": self.get_quick_stats(df),
            "columns": self.get_column_info(df),
            "distributions": {},
            "correlations": None,
            "missing_matrix": {},
            "outliers": {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Distributions for numeric columns
        for col in numeric_cols[:15]:  # Limit to 15 columns
            try:
                clean = df[col].dropna()
                if len(clean) == 0 or clean.nunique() < 1:
                    continue
                
                values = clean.tolist()
                if len(values) > 500:
                    values = pd.Series(values).sample(500, random_state=42).tolist()
                
                n_bins = max(1, min(30, int(clean.nunique())))
                hist, bin_edges = np.histogram(clean, bins=n_bins)
                result["distributions"][col] = {
                    "histogram": {
                        "counts": [int(c) for c in hist.tolist()],
                        "edges": [self._safe_float(e) for e in bin_edges.tolist()]
                    },
                    "values": [self._safe_float(v) for v in values[:200]]
                }
            except Exception:
                pass
        
        # Correlation matrix for numeric columns
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr()
                # Replace NaN/inf with 0 for JSON serialization
                corr = corr.fillna(0)
                result["correlations"] = {
                    "columns": numeric_cols,
                    "matrix": [[self._safe_float(v) for v in row] for row in corr.values.tolist()]
                }
            except Exception:
                pass
        
        # Missing value patterns
        missing = df.isnull().sum()
        result["missing_matrix"] = {
            "columns": list(missing.index),
            "counts": [int(v) for v in missing.values],
            "percentages": [self._safe_float(v / len(df) * 100) if len(df) > 0 else 0 for v in missing.values]
        }
        
        # Outlier detection using IQR
        for col in numeric_cols[:10]:
            try:
                clean = df[col].dropna()
                if len(clean) == 0:
                    continue
                Q1 = float(clean.quantile(0.25))
                Q3 = float(clean.quantile(0.75))
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_count = int(((clean < lower) | (clean > upper)).sum())
                result["outliers"][col] = {
                    "count": outlier_count,
                    "percentage": self._safe_float(outlier_count / len(df) * 100),
                    "lower_bound": self._safe_float(lower),
                    "upper_bound": self._safe_float(upper)
                }
            except Exception:
                pass
        
        return result
    
    def apply_transform(self, df, step):
        """Apply a preprocessing transformation step"""
        action = step.get("action")
        
        if action == "drop_missing":
            columns = step.get("columns", df.columns.tolist())
            df = df.dropna(subset=columns)
        
        elif action == "fill_missing_mean":
            columns = step.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
            for col in columns:
                if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].mean())
        
        elif action == "fill_missing_median":
            columns = step.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
            for col in columns:
                if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
        
        elif action == "fill_missing_mode":
            columns = step.get("columns", df.columns.tolist())
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")
        
        elif action == "drop_duplicates":
            df = df.drop_duplicates()
        
        elif action == "drop_columns":
            columns = step.get("columns", [])
            df = df.drop(columns=[c for c in columns if c in df.columns])
        
        elif action == "encode_label":
            columns = step.get("columns", df.select_dtypes(include=["object"]).columns.tolist())
            for col in columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        
        elif action == "scale_standard":
            columns = step.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
            scaler = StandardScaler()
            for col in columns:
                if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                    df[col] = scaler.fit_transform(df[[col]])
        
        elif action == "scale_minmax":
            columns = step.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
            scaler = MinMaxScaler()
            for col in columns:
                if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                    df[col] = scaler.fit_transform(df[[col]])
        
        elif action == "remove_outliers":
            columns = step.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
            for col in columns:
                if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        
        return df
