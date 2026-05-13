"""
DataForge — ML Experimentation Platform
FastAPI Backend Entry Point
"""

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import os
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

from backend.core.data_processor import DataProcessor
from backend.core.ml_engine import MLEngine
from backend.core.experiment_tracker import ExperimentTracker

app = FastAPI(title="DataForge", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores
datasets = {}
experiments_store = {}
trained_models = {}
training_jobs = {}

# Initialize core engines
data_processor = DataProcessor()
ml_engine = MLEngine()
experiment_tracker = ExperimentTracker()

# Serve static frontend files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Sample datasets directory
SAMPLE_DATA_DIR = Path(__file__).parent.parent / "sample_data"
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_DATA_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def serve_app():
    """Serve the main SPA"""
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(str(index_path))


# ==================== DATA ENDPOINTS ====================

@app.post("/api/data/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and parse a CSV/Excel dataset"""
    try:
        dataset_id = str(uuid.uuid4())[:8]
        content = await file.read()
        
        # Save file
        file_path = UPLOAD_DIR / f"{dataset_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Parse dataset
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        # Store dataset
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": file.filename,
            "df": df,
            "uploaded_at": datetime.now().isoformat(),
            "file_path": str(file_path)
        }
        
        # Generate quick stats
        stats = data_processor.get_quick_stats(df)
        
        return {
            "id": dataset_id,
            "name": file.filename,
            "stats": stats,
            "message": f"Dataset uploaded successfully! {len(df)} rows × {len(df.columns)} columns"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/samples")
async def list_sample_datasets():
    """List available sample datasets"""
    samples = data_processor.get_sample_datasets()
    return {"datasets": samples}


@app.post("/api/data/samples/{name}")
async def load_sample_dataset(name: str):
    """Load a built-in sample dataset"""
    try:
        df, description = data_processor.load_sample_dataset(name)
        dataset_id = str(uuid.uuid4())[:8]
        
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": name,
            "df": df,
            "uploaded_at": datetime.now().isoformat(),
            "description": description
        }
        
        stats = data_processor.get_quick_stats(df)
        
        return {
            "id": dataset_id,
            "name": name,
            "description": description,
            "stats": stats,
            "message": f"Sample dataset '{name}' loaded! {len(df)} rows × {len(df.columns)} columns"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/{dataset_id}/eda")
async def get_eda(dataset_id: str):
    """Get full auto-EDA for a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["df"]
    eda = data_processor.full_eda(df)
    return eda


@app.get("/api/data/{dataset_id}/columns")
async def get_columns(dataset_id: str):
    """Get column information for a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["df"]
    columns = data_processor.get_column_info(df)
    return {"columns": columns}


@app.get("/api/data/{dataset_id}/preview")
async def preview_data(dataset_id: str, rows: int = Query(default=20, le=100)):
    """Get a preview of the dataset rows"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["df"]
    preview = df.head(rows).fillna("null")
    return {
        "columns": list(df.columns),
        "data": preview.values.tolist(),
        "total_rows": len(df)
    }


# ==================== MODEL / TRAINING ENDPOINTS ====================

@app.post("/api/models/train")
async def train_model(config: dict):
    """Train a model with given configuration"""
    try:
        dataset_id = config.get("dataset_id")
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = datasets[dataset_id]["df"]
        
        result = ml_engine.train_model(
            df=df,
            target_column=config.get("target_column"),
            feature_columns=config.get("feature_columns"),
            model_type=config.get("model_type", "random_forest"),
            problem_type=config.get("problem_type", "classification"),
            test_size=config.get("test_size", 0.2),
            hyperparams=config.get("hyperparams", {})
        )
        
        # Store model and create experiment
        model_id = str(uuid.uuid4())[:8]
        trained_models[model_id] = result["model_obj"]
        
        experiment = experiment_tracker.log_experiment(
            model_id=model_id,
            dataset_name=datasets[dataset_id]["name"],
            model_type=config.get("model_type", "random_forest"),
            problem_type=config.get("problem_type", "classification"),
            metrics=result["metrics"],
            params=config.get("hyperparams", {}),
            feature_columns=config.get("feature_columns", []),
            target_column=config.get("target_column", "")
        )
        
        return {
            "model_id": model_id,
            "experiment": experiment,
            "metrics": result["metrics"],
            "feature_importance": result.get("feature_importance", []),
            "confusion_matrix": result.get("confusion_matrix"),
            "classification_report": result.get("classification_report"),
            "predictions_sample": result.get("predictions_sample", []),
            "message": f"Model trained successfully! Model ID: {model_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/compare")
async def compare_models(config: dict):
    """Train and compare multiple models"""
    try:
        dataset_id = config.get("dataset_id")
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = datasets[dataset_id]["df"]
        model_types = config.get("model_types", ["random_forest", "logistic_regression"])
        
        results = ml_engine.compare_models(
            df=df,
            target_column=config.get("target_column"),
            feature_columns=config.get("feature_columns"),
            model_types=model_types,
            problem_type=config.get("problem_type", "classification"),
            test_size=config.get("test_size", 0.2)
        )
        
        # Log each model as experiment
        for r in results:
            model_id = str(uuid.uuid4())[:8]
            trained_models[model_id] = r.pop("model_obj", None)
            r["model_id"] = model_id
            experiment_tracker.log_experiment(
                model_id=model_id,
                dataset_name=datasets[dataset_id]["name"],
                model_type=r["model_type"],
                problem_type=config.get("problem_type", "classification"),
                metrics=r["metrics"],
                params={},
                feature_columns=config.get("feature_columns", []),
                target_column=config.get("target_column", "")
            )
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EXPERIMENT ENDPOINTS ====================

@app.get("/api/experiments")
async def list_experiments():
    """List all experiments"""
    return {"experiments": experiment_tracker.get_all_experiments()}


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get details of a specific experiment"""
    exp = experiment_tracker.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@app.delete("/api/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment"""
    success = experiment_tracker.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment deleted"}


# ==================== PIPELINE ENDPOINTS ====================

@app.post("/api/pipeline/execute")
async def execute_pipeline(config: dict):
    """Execute a complete ML pipeline"""
    try:
        dataset_id = config.get("dataset_id")
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = datasets[dataset_id]["df"].copy()
        steps = config.get("steps", [])
        
        # Execute preprocessing steps
        for step in steps:
            df = data_processor.apply_transform(df, step)
        
        # Store the transformed dataset
        new_id = str(uuid.uuid4())[:8]
        datasets[new_id] = {
            "id": new_id,
            "name": f"{datasets[dataset_id]['name']} (transformed)",
            "df": df,
            "uploaded_at": datetime.now().isoformat()
        }
        
        stats = data_processor.get_quick_stats(df)
        
        return {
            "dataset_id": new_id,
            "stats": stats,
            "rows": len(df),
            "columns": len(df.columns),
            "message": "Pipeline executed successfully!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== COPILOT ENDPOINTS ====================

@app.post("/api/copilot/chat")
async def copilot_chat(request: dict):
    """AI Copilot chat — provides intelligent ML guidance"""
    message = request.get("message", "")
    context = request.get("context", {})
    
    # Get dataset context if available
    dataset_id = context.get("dataset_id")
    dataset_info = None
    column_details = None
    sample_rows = None
    
    if dataset_id and dataset_id in datasets:
        df = datasets[dataset_id]["df"]
        dataset_info = data_processor.get_quick_stats(df)
        dataset_info["name"] = datasets[dataset_id].get("name", "unknown")
        
        # Provide detailed column info for context
        column_details = data_processor.get_column_info(df)
        
        # Provide a few sample rows as context (as dicts)
        try:
            preview_df = df.head(5).fillna("null")
            sample_rows = preview_df.to_dict(orient="records")
            # Convert numpy types to native python for JSON
            for row in sample_rows:
                for k, v in row.items():
                    if isinstance(v, (np.integer,)):
                        row[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        row[k] = float(v)
                    elif isinstance(v, np.bool_):
                        row[k] = bool(v)
        except Exception:
            sample_rows = None
    
    # Get experiment context
    recent_experiments = experiment_tracker.get_all_experiments()[-5:]
    
    # Generate intelligent response
    from backend.core.ai_copilot import AICopilot
    copilot = AICopilot()
    response = copilot.generate_response(
        message=message,
        dataset_info=dataset_info,
        column_details=column_details,
        sample_rows=sample_rows,
        experiments=recent_experiments
    )
    
    return {"response": response}


# ==================== DASHBOARD ENDPOINTS ====================

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard overview stats"""
    all_experiments = experiment_tracker.get_all_experiments()
    
    best_accuracy = 0
    if all_experiments:
        for exp in all_experiments:
            acc = exp.get("metrics", {}).get("accuracy", 0)
            if acc > best_accuracy:
                best_accuracy = acc
    
    return {
        "total_datasets": len(datasets),
        "total_experiments": len(all_experiments),
        "total_models": len(trained_models),
        "best_accuracy": round(best_accuracy, 4),
        "recent_experiments": all_experiments[-5:] if all_experiments else []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
