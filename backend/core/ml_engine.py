"""
ML Engine — Handles model training, evaluation, and comparison
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import time

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


class MLEngine:
    
    CLASSIFIERS = {
        "logistic_regression": ("Logistic Regression", LogisticRegression, {"max_iter": 1000, "random_state": 42}),
        "decision_tree": ("Decision Tree", DecisionTreeClassifier, {"random_state": 42}),
        "random_forest": ("Random Forest", RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
        "gradient_boosting": ("Gradient Boosting", GradientBoostingClassifier, {"n_estimators": 100, "random_state": 42}),
        "adaboost": ("AdaBoost", AdaBoostClassifier, {"n_estimators": 100, "random_state": 42}),
        "svm": ("Support Vector Machine", SVC, {"probability": True, "random_state": 42}),
        "knn": ("K-Nearest Neighbors", KNeighborsClassifier, {"n_neighbors": 5}),
        "naive_bayes": ("Naive Bayes", GaussianNB, {}),
    }
    
    REGRESSORS = {
        "linear_regression": ("Linear Regression", LinearRegression, {}),
        "ridge": ("Ridge Regression", Ridge, {"random_state": 42}),
        "lasso": ("Lasso Regression", Lasso, {"random_state": 42}),
        "decision_tree": ("Decision Tree", DecisionTreeRegressor, {"random_state": 42}),
        "random_forest": ("Random Forest", RandomForestRegressor, {"n_estimators": 100, "random_state": 42}),
        "gradient_boosting": ("Gradient Boosting", GradientBoostingRegressor, {"n_estimators": 100, "random_state": 42}),
        "adaboost": ("AdaBoost", AdaBoostRegressor, {"n_estimators": 100, "random_state": 42}),
        "svr": ("Support Vector Regressor", SVR, {}),
        "knn": ("K-Nearest Neighbors", KNeighborsRegressor, {"n_neighbors": 5}),
    }
    
    def __init__(self):
        if HAS_XGBOOST:
            self.CLASSIFIERS["xgboost"] = ("XGBoost", XGBClassifier, {"n_estimators": 100, "random_state": 42, "use_label_encoder": False, "eval_metric": "logloss"})
            self.REGRESSORS["xgboost"] = ("XGBoost", XGBRegressor, {"n_estimators": 100, "random_state": 42})
        if HAS_LIGHTGBM:
            self.CLASSIFIERS["lightgbm"] = ("LightGBM", LGBMClassifier, {"n_estimators": 100, "random_state": 42, "verbose": -1})
            self.REGRESSORS["lightgbm"] = ("LightGBM", LGBMRegressor, {"n_estimators": 100, "random_state": 42, "verbose": -1})
    
    def _prepare_data(self, df, target_column, feature_columns, problem_type):
        """Prepare data for training"""
        if feature_columns:
            X = df[feature_columns].copy()
        else:
            X = df.drop(columns=[target_column]).copy()
        
        y = df[target_column].copy()
        
        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target if classification
        target_le = None
        if problem_type == "classification":
            if y.dtype in ("object", "category"):
                target_le = LabelEncoder()
                y = pd.Series(target_le.fit_transform(y.astype(str)))
            elif y.dtype.kind == 'f':
                # Continuous float target — convert to int if values are whole numbers
                if (y.dropna() == y.dropna().astype(int)).all():
                    y = y.astype(int)
                else:
                    # Truly continuous — bin it or raise a helpful error
                    target_le = LabelEncoder()
                    y = pd.Series(target_le.fit_transform(y.astype(str)))
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "Unknown")
        
        # Scale numeric features
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        return X, y, target_le, list(X.columns)
    
    def train_model(self, df, target_column, feature_columns, model_type, problem_type, test_size=0.2, hyperparams=None):
        """Train a single model and return results"""
        X, y, target_le, used_features = self._prepare_data(df, target_column, feature_columns, problem_type)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Get model class and default params
        if problem_type == "classification":
            model_info = self.CLASSIFIERS.get(model_type)
        else:
            model_info = self.REGRESSORS.get(model_type)
        
        if not model_info:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_name, model_class, default_params = model_info
        
        # Merge default params with user hyperparams
        params = {**default_params}
        if hyperparams:
            params.update(hyperparams)
        
        # Train
        start_time = time.time()
        model = model_class(**params)
        model.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 3)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if problem_type == "classification":
            metrics = self._classification_metrics(y_test, y_pred, model, X_test)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            # Classification report
            try:
                target_names = list(target_le.classes_) if target_le else [str(c) for c in sorted(y.unique())]
                cr = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                # Convert numpy types for JSON serialization
                cr_clean = {}
                for k, v in cr.items():
                    if isinstance(v, dict):
                        cr_clean[k] = {kk: round(float(vv), 4) for kk, vv in v.items()}
                    else:
                        cr_clean[k] = round(float(v), 4)
            except Exception:
                cr_clean = None
        else:
            metrics = self._regression_metrics(y_test, y_pred)
            cm = None
            cr_clean = None
        
        metrics["training_time"] = training_time
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, used_features)
        
        # Predictions sample
        pred_sample = []
        for i in range(min(10, len(y_test))):
            actual = float(y_test.iloc[i]) if hasattr(y_test, 'iloc') else float(y_test[i])
            predicted = float(y_pred[i])
            if target_le:
                actual = str(target_le.inverse_transform([int(actual)])[0])
                predicted = str(target_le.inverse_transform([int(predicted)])[0])
            pred_sample.append({"actual": actual, "predicted": predicted})
        
        # Cross-validation score
        try:
            cv_model = model_class(**params)
            cv_scores = cross_val_score(cv_model, X, y, cv=min(5, len(X)), scoring="accuracy" if problem_type == "classification" else "r2")
            metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
            metrics["cv_std"] = round(float(cv_scores.std()), 4)
        except Exception:
            metrics["cv_mean"] = None
            metrics["cv_std"] = None
        
        return {
            "model_name": model_name,
            "model_type": model_type,
            "model_obj": model,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "confusion_matrix": cm,
            "classification_report": cr_clean,
            "predictions_sample": pred_sample,
            "features_used": used_features
        }
    
    def compare_models(self, df, target_column, feature_columns, model_types, problem_type, test_size=0.2):
        """Train and compare multiple models"""
        results = []
        
        for model_type in model_types:
            try:
                result = self.train_model(
                    df=df,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    model_type=model_type,
                    problem_type=problem_type,
                    test_size=test_size
                )
                results.append(result)
            except Exception as e:
                results.append({
                    "model_type": model_type,
                    "model_name": model_type,
                    "error": str(e),
                    "metrics": {}
                })
        
        # Sort by primary metric
        if problem_type == "classification":
            results.sort(key=lambda x: x.get("metrics", {}).get("accuracy", 0), reverse=True)
        else:
            results.sort(key=lambda x: x.get("metrics", {}).get("r2", 0), reverse=True)
        
        return results
    
    def _classification_metrics(self, y_true, y_pred, model, X_test):
        """Calculate classification metrics"""
        avg = "weighted" if len(np.unique(y_true)) > 2 else "binary"
        
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        }
        
        # AUC-ROC
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
                else:
                    metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")), 4)
        except Exception:
            metrics["auc_roc"] = None
        
        return metrics
    
    def _regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }
    
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from model"""
        importance = []
        
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
                if len(importances) != len(feature_names):
                    return []
            else:
                return []
            
            for name, imp in zip(feature_names, importances):
                importance.append({
                    "feature": name,
                    "importance": round(float(imp), 6)
                })
            
            importance.sort(key=lambda x: x["importance"], reverse=True)
        except Exception:
            pass
        
        return importance
    
    def get_available_models(self, problem_type):
        """Return list of available models"""
        if problem_type == "classification":
            return [
                {"id": k, "name": v[0], "description": self._model_description(k, "classification")}
                for k, v in self.CLASSIFIERS.items()
            ]
        else:
            return [
                {"id": k, "name": v[0], "description": self._model_description(k, "regression")}
                for k, v in self.REGRESSORS.items()
            ]
    
    def _model_description(self, model_type, problem_type):
        """Return a beginner-friendly description"""
        descriptions = {
            "logistic_regression": "Simple and interpretable. Great starting point for binary classification.",
            "linear_regression": "Finds linear relationships. Good baseline for regression.",
            "ridge": "Linear regression with regularization. Prevents overfitting.",
            "lasso": "Linear regression that can select important features automatically.",
            "decision_tree": "Makes decisions like a flowchart. Easy to understand and visualize.",
            "random_forest": "Combines many decision trees for better accuracy. Very reliable.",
            "gradient_boosting": "Builds trees sequentially, each fixing previous mistakes. Very powerful.",
            "adaboost": "Focuses on hard-to-classify examples. Good for imbalanced data.",
            "svm": "Finds the best boundary between classes. Works well in high dimensions.",
            "svr": "Support vector machine for regression. Good with non-linear data.",
            "knn": "Classifies based on nearest neighbors. Simple and intuitive.",
            "naive_bayes": "Fast and efficient. Works surprisingly well on many problems.",
            "xgboost": "State-of-the-art gradient boosting. Often wins competitions.",
            "lightgbm": "Fast gradient boosting for large datasets. Very efficient.",
        }
        return descriptions.get(model_type, "A machine learning model.")
