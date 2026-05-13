"""
AI Copilot — Powered by Groq LLM (Llama 3) with intelligent fallback
"""

import os
import json
import httpx

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are DataForge AI Copilot — an expert ML assistant embedded in a machine learning experimentation platform.

Your personality:
- Friendly, encouraging, and beginner-friendly
- Use emojis to make responses engaging
- Give practical, actionable advice
- When explaining concepts, use analogies and simple language
- Always structure responses with headers, bullet points, and tables when appropriate
- Use markdown formatting (headers, bold, code blocks, tables)

Your capabilities:
- Recommend ML models based on dataset characteristics
- Diagnose model problems (overfitting, underfitting, poor metrics)
- Explain ML concepts in simple terms
- Suggest feature engineering techniques
- Guide preprocessing decisions
- Interpret metrics and results
- Help with hyperparameter tuning
- Analyze specific columns and suggest how to handle them

When given dataset context, personalize your advice based on:
- Dataset name and what it contains
- Column names, types, and statistics (mean, std, min, max, unique values)
- Sample data rows to understand the actual values
- Data quality score and missing values
- Problem type (classification/regression)

When given experiment context, reference actual metrics and suggest improvements.

IMPORTANT: When dataset context is provided, ALWAYS reference specific column names, data types, and values from the dataset. Give advice tailored to THIS specific dataset, not generic advice. For example, if you see a 'Sex' column with values 'male'/'female', recommend encoding it. If you see high cardinality, suggest appropriate handling.

Keep responses concise but comprehensive. Use markdown tables for comparisons.
"""


class AICopilot:

    def generate_response(self, message, dataset_info=None, column_details=None, sample_rows=None, experiments=None):
        """Generate response using Groq LLM with fallback to rules"""
        # Build context
        context_parts = []
        if dataset_info:
            context_parts.append(
                f"**Dataset:** \"{dataset_info.get('name', 'unknown')}\"\n"
                f"- Rows: {dataset_info.get('rows', '?')}, Columns: {dataset_info.get('columns', '?')}\n"
                f"- Numeric columns: {dataset_info.get('numeric_columns', '?')}, Categorical: {dataset_info.get('categorical_columns', '?')}\n"
                f"- Quality score: {dataset_info.get('quality_score', '?')}/100\n"
                f"- Missing values: {dataset_info.get('missing_percentage', 0)}% ({dataset_info.get('missing_values', 0)} total)\n"
                f"- Duplicate rows: {dataset_info.get('duplicate_rows', 0)} ({dataset_info.get('duplicate_percentage', 0)}%)\n"
                f"- Memory: {dataset_info.get('memory_usage_mb', '?')} MB\n"
                f"- All column names: {', '.join(dataset_info.get('column_names', []))}"
            )

        if column_details:
            col_summaries = []
            for col in column_details:
                summary = f"  - **{col['name']}**: {'numeric' if col.get('is_numeric') else 'categorical'}, dtype={col.get('dtype')}, {col.get('unique')} unique ({col.get('unique_pct', 0)}%), missing={col.get('missing', 0)} ({col.get('missing_pct', 0)}%)"
                if col.get('is_numeric'):
                    summary += f", mean={col.get('mean')}, std={col.get('std')}, min={col.get('min')}, max={col.get('max')}, skewness={col.get('skewness')}"
                else:
                    top_vals = col.get('top_values', {})
                    if top_vals:
                        top_str = ', '.join(f"'{k}'({v})" for k, v in list(top_vals.items())[:5])
                        summary += f", top values: {top_str}"
                col_summaries.append(summary)
            context_parts.append("**Column Details:**\n" + "\n".join(col_summaries))

        if sample_rows:
            try:
                sample_str = json.dumps(sample_rows[:3], indent=2, default=str)
                context_parts.append(f"**Sample Rows (first 3):**\n```json\n{sample_str}\n```")
            except Exception:
                pass

        if experiments:
            exp_summary = []
            for exp in experiments[:5]:
                m = exp.get("metrics", {})
                if exp.get("problem_type") == "classification":
                    exp_summary.append(f"- {exp.get('model_type')}: accuracy={m.get('accuracy','?')}, f1={m.get('f1','?')}, precision={m.get('precision','?')}, recall={m.get('recall','?')}")
                else:
                    exp_summary.append(f"- {exp.get('model_type')}: r2={m.get('r2','?')}, rmse={m.get('rmse','?')}, mae={m.get('mae','?')}")
            context_parts.append("**Recent experiments:**\n" + "\n".join(exp_summary))

        # Try Groq LLM first
        api_key = GROQ_API_KEY
        if api_key and len(api_key) > 10:
            try:
                return self._call_groq(message, context_parts, api_key)
            except Exception as e:
                print(f"Groq API error: {e}, falling back to rules")

        # Fallback to rule-based
        return self._rule_based_response(message, dataset_info, column_details, sample_rows, experiments, context_parts)

    def _call_groq(self, message, context_parts, api_key):
        """Call Groq API for LLM response"""
        user_message = message
        if context_parts:
            user_message += "\n\n---\n[Platform Context — Use this to give specific advice]\n" + "\n\n".join(context_parts)

        response = httpx.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.9
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _rule_based_response(self, message, dataset_info, column_details, sample_rows, experiments, context_parts):
        """Fallback rule-based responses"""
        message_lower = message.lower().strip()

        response = self._match_intent(message_lower, dataset_info, column_details, sample_rows, experiments)

        if context_parts:
            response += "\n\n---\n📊 **Context I'm using:**\n" + "\n".join(f"• {p}" for p in context_parts)

        return response

    def _match_intent(self, message, dataset_info, column_details, sample_rows, experiments):
        """Match user intent and generate appropriate response"""

        if any(w in message for w in ["hello", "hi", "hey", "help", "what can you do"]):
            return self._greeting(dataset_info)

        if any(w in message for w in ["overfit", "overfitting", "high variance"]):
            return self._overfitting(dataset_info)

        if any(w in message for w in ["underfit", "underfitting", "high bias", "low accuracy"]):
            return self._underfitting()

        if any(w in message for w in ["which model", "best model", "recommend", "suggest model"]):
            return self._recommend_model(dataset_info, column_details)

        if any(w in message for w in ["feature", "engineering"]):
            return self._feature_engineering(dataset_info, column_details)

        if any(w in message for w in ["missing", "null", "nan", "impute"]):
            return self._missing_data(dataset_info, column_details)

        if any(w in message for w in ["imbalance", "imbalanced"]):
            return self._imbalanced()

        if any(w in message for w in ["hyperparameter", "tuning", "tune"]):
            return self._hyperparameter()

        if any(w in message for w in ["metric", "accuracy", "precision", "recall", "f1"]):
            return self._metrics(experiments)

        if any(w in message for w in ["start", "begin", "getting started", "how to"]):
            return self._getting_started(dataset_info, column_details)

        if any(w in message for w in ["preprocess", "clean", "scale", "normalize"]):
            return self._preprocessing(dataset_info, column_details)

        if any(w in message for w in ["describe", "about", "tell me about", "what is this", "summary", "analyze"]):
            return self._describe_dataset(dataset_info, column_details, sample_rows)

        return self._default(dataset_info)

    def _greeting(self, info):
        msg = """# 👋 Hi! I'm your ML Copilot

I help you build better models. Ask me:
- **"Which model should I use?"** → Personalized recommendations
- **"My model is overfitting"** → Diagnosis & solutions
- **"Feature engineering tips"** → Improve predictions
- **"How to get started?"** → Step-by-step guide
- **"Describe my dataset"** → Dataset analysis"""

        if info:
            msg += f"\n\n✅ I can see your **{info.get('name', 'current')}** dataset ({info.get('rows', '?')} rows × {info.get('columns', '?')} columns). Ask me anything about it!"
        else:
            msg += "\n\n💡 Load a dataset first for personalized advice!"
        return msg

    def _describe_dataset(self, info, column_details, sample_rows):
        """Provide a detailed description of the current dataset"""
        if not info:
            return "📂 **No dataset loaded!** Go to Data Explorer and upload a CSV or pick a sample dataset first."

        msg = f"# 📊 Dataset Analysis: {info.get('name', 'Your Dataset')}\n\n"
        msg += f"**Overview:** {info.get('rows', '?')} rows × {info.get('columns', '?')} columns\n"
        msg += f"- 🔢 Numeric columns: {info.get('numeric_columns', '?')}\n"
        msg += f"- 🏷️ Categorical columns: {info.get('categorical_columns', '?')}\n"
        msg += f"- ✨ Quality score: {info.get('quality_score', '?')}/100\n"
        msg += f"- ❓ Missing values: {info.get('missing_percentage', 0)}%\n"
        msg += f"- 🔁 Duplicates: {info.get('duplicate_rows', 0)} rows\n\n"

        if column_details:
            msg += "## Column Breakdown\n\n"
            msg += "| Column | Type | Unique | Missing | Key Stats |\n"
            msg += "|--------|------|--------|---------|----------|\n"
            for col in column_details:
                col_type = "Numeric" if col.get("is_numeric") else "Categorical"
                stats = ""
                if col.get("is_numeric"):
                    stats = f"μ={col.get('mean')}, σ={col.get('std')}"
                else:
                    top = col.get("top_values", {})
                    if top:
                        stats = ", ".join(f"{k}({v})" for k, v in list(top.items())[:2])
                msg += f"| {col['name']} | {col_type} | {col.get('unique')} | {col.get('missing', 0)} | {stats} |\n"

        msg += "\n## 💡 Recommendations\n\n"
        if info.get('missing_percentage', 0) > 0:
            msg += f"- ⚠️ **Handle missing values** ({info['missing_percentage']}%) — try Pipeline Builder\n"
        if info.get('duplicate_percentage', 0) > 5:
            msg += f"- ⚠️ **Remove duplicates** ({info['duplicate_percentage']}%)\n"
        if info.get('categorical_columns', 0) > 0:
            msg += "- 🏷️ **Encode categorical columns** before training (Pipeline Builder → Label Encode)\n"
        msg += "- 🚀 **Ready to train?** Head to Model Arena and select your target column!\n"

        return msg

    def _overfitting(self, info):
        r = """# 🔧 Fixing Overfitting

## Quick Fixes
1. **Add regularization** (increase L1/L2 penalty)
2. **Simplify model** (fewer trees, lower depth)
3. **Use cross-validation** (5-fold CV)
4. **Try Random Forest** (naturally resists overfitting)

## Advanced
- Feature selection (remove unimportant features)
- Early stopping (for boosting models)
- Get more training data"""
        if info and info.get("rows", 0) < 500:
            r += f"\n\n⚠️ With only {info['rows']} rows, overfitting is very likely. Use simpler models."
        return r

    def _underfitting(self):
        return """# 📈 Fixing Underfitting

## Quick Fixes
1. **Try more complex models**: Random Forest → XGBoost
2. **Create new features**: Interactions, polynomials
3. **Reduce regularization**
4. **Increase `n_estimators`** for boosting models

## Check
- Are features properly encoded?
- Is the data quality good?"""

    def _recommend_model(self, info, column_details):
        if info:
            rows = info.get("rows", 0)
            cat_cols = info.get("categorical_columns", 0)
            num_cols = info.get("numeric_columns", 0)

            msg = ""
            if rows < 500:
                msg = """# 🎯 Model Recommendations (Small Dataset)

| Pick | Model | Why |
|------|-------|-----|
| 🥇 | **Logistic Regression** | Won't overfit, interpretable |
| 🥈 | **Random Forest** | Reliable, handles mixed types |
| 🥉 | **KNN** | Simple, no assumptions |

⚠️ Avoid complex models (XGBoost, Neural Nets) with small data."""
            else:
                msg = """# 🎯 Model Recommendations

| Pick | Model | Why |
|------|-------|-----|
| 🥇 | **XGBoost** | State-of-the-art accuracy |
| 🥈 | **Random Forest** | Reliable all-rounder |
| 🥉 | **Gradient Boosting** | Great with tuning |

💡 Start with Random Forest, then try XGBoost."""

            if cat_cols > 0 and column_details:
                cat_names = [c["name"] for c in column_details if not c.get("is_numeric")]
                if cat_names:
                    msg += f"\n\n⚠️ You have categorical columns ({', '.join(cat_names[:5])}). Make sure to **encode them** first in Pipeline Builder, or use tree-based models that handle them internally."
            return msg

        return """# 🎯 Model Recommendations

**Classification:** Random Forest → XGBoost → Logistic Regression
**Regression:** Random Forest → Gradient Boosting → Ridge

💡 Load data for personalized recommendations!"""

    def _feature_engineering(self, info, column_details):
        msg = """# ✨ Feature Engineering

## Numeric: Log transform, polynomial features, binning
## Categorical: One-hot encode, label encode, target encode
## DateTime: Extract year, month, day_of_week, is_weekend
## Create: Ratios, interactions, boolean flags"""

        if column_details:
            cat_cols = [c for c in column_details if not c.get("is_numeric")]
            high_card = [c for c in cat_cols if c.get("unique", 0) > 20]
            if high_card:
                msg += f"\n\n⚠️ **High cardinality alert:** {', '.join(c['name'] for c in high_card)} have many unique values. Consider target encoding instead of one-hot."
            num_cols = [c for c in column_details if c.get("is_numeric") and abs(c.get("skewness", 0) or 0) > 2]
            if num_cols:
                msg += f"\n\n📊 **Skewed features:** {', '.join(c['name'] for c in num_cols)} — try log transform to normalize."

        return msg

    def _missing_data(self, info, column_details):
        r = """# 🔍 Handling Missing Data

| Missing % | Action |
|-----------|--------|
| >50% | Drop the column |
| >20% | KNN/iterative imputation |
| <20% | Median (numeric) or Mode (categorical) |
| <5% | Simple fill or drop rows |"""

        if column_details:
            missing_cols = [c for c in column_details if c.get("missing", 0) > 0]
            if missing_cols:
                r += "\n\n## 🔍 In Your Dataset:\n"
                for col in sorted(missing_cols, key=lambda c: c.get("missing_pct", 0), reverse=True):
                    pct = col.get("missing_pct", 0)
                    suggestion = "Drop column" if pct > 50 else ("KNN imputer" if pct > 20 else ("Median/mode fill" if pct > 5 else "Drop rows or simple fill"))
                    r += f"- **{col['name']}**: {pct}% missing → {suggestion}\n"
            else:
                r += "\n\n✅ **Great news!** Your dataset has no missing values."
        elif info and info.get("missing_percentage", 0) > 0:
            r += f"\n\nYour data: **{info['missing_percentage']}% missing** → {'Use median/mode' if info['missing_percentage'] < 10 else 'Consider KNN imputer'}"
        return r

    def _imbalanced(self):
        return """# ⚖️ Imbalanced Data

1. Use `class_weight='balanced'`
2. Switch metric to **F1-score** or **AUC-ROC**
3. Try **SMOTE** for synthetic oversampling
4. Use **stratified splits**"""

    def _hyperparameter(self):
        return """# 🎛️ Hyperparameter Tuning

**Random Forest:** `n_estimators`(100-500), `max_depth`(5-30)
**XGBoost:** `learning_rate`(0.01-0.3), `max_depth`(3-10)
**LogReg:** `C`(0.001-100)

Strategy: Default → Random Search → Fine-tune"""

    def _metrics(self, experiments):
        msg = """# 📊 ML Metrics

| Metric | Use When |
|--------|----------|
| **Accuracy** | Balanced classes |
| **F1-Score** | Imbalanced data |
| **AUC-ROC** | Comparing models |
| **R²** | Regression quality |
| **RMSE** | Penalize big errors |"""

        if experiments:
            msg += "\n\n## Your Recent Results:\n"
            for exp in experiments[:3]:
                m = exp.get("metrics", {})
                if exp.get("problem_type") == "classification":
                    msg += f"- **{exp.get('model_type')}**: accuracy={m.get('accuracy','?')}, f1={m.get('f1','?')}\n"
                else:
                    msg += f"- **{exp.get('model_type')}**: R²={m.get('r2','?')}, RMSE={m.get('rmse','?')}\n"
        return msg

    def _getting_started(self, info, column_details):
        msg = """# 🚀 Getting Started

1. **📂 Data Explorer** → Upload CSV or pick a sample
2. **🔍 Auto-EDA** → Understand your data
3. **⚙️ Pipeline** → Clean & preprocess
4. **⚔️ Arena** → Train & compare models
5. **🧪 Experiments** → Track all results"""

        if info:
            msg += f"\n\n✅ **You already have data loaded!** ({info.get('name', 'dataset')}: {info.get('rows', '?')} rows)\n\n"
            msg += "**Suggested next steps:**\n"
            if info.get("missing_percentage", 0) > 0:
                msg += f"1. ⚙️ Handle {info['missing_percentage']}% missing data in **Pipeline Builder**\n"
            if info.get("categorical_columns", 0) > 0:
                msg += f"2. 🏷️ Encode {info['categorical_columns']} categorical columns\n"
            msg += "3. ⚔️ Head to **Model Arena** and start training!\n"
        else:
            msg += "\n\n💡 Start with **Iris dataset** + **Random Forest** = 95%+ accuracy!"
        return msg

    def _preprocessing(self, info, column_details):
        msg = """# 🧹 Preprocessing Checklist

1. ✅ Remove duplicates
2. ✅ Handle missing values
3. ✅ Encode categoricals
4. ✅ Scale features (for SVM, KNN, LogReg)
5. ✅ Remove outliers (optional)

💡 Tree models (RF, XGBoost) don't need scaling!"""

        if column_details:
            cat_cols = [c["name"] for c in column_details if not c.get("is_numeric")]
            if cat_cols:
                msg += f"\n\n🏷️ **Categorical columns to encode:** {', '.join(cat_cols)}"
            missing_cols = [c["name"] for c in column_details if c.get("missing", 0) > 0]
            if missing_cols:
                msg += f"\n❓ **Columns with missing values:** {', '.join(missing_cols)}"
        return msg

    def _default(self, info):
        msg = """# 🤔 I can help with:

- **"Which model?"** → Recommendations
- **"Overfitting"** → Solutions
- **"Missing data"** → Handling strategies
- **"Get started"** → Step-by-step guide
- **"Feature engineering"** → Improve features
- **"Explain metrics"** → Understand results
- **"Describe my dataset"** → Dataset analysis"""

        if info:
            msg += f"\n\n📊 I have context about your **{info.get('name', 'current')}** dataset. Ask me something specific!"
        return msg
