"""
Experiment Tracker — Logs and manages experiment runs
"""

import uuid
from datetime import datetime


class ExperimentTracker:
    
    def __init__(self):
        self.experiments = {}
    
    def log_experiment(self, model_id, dataset_name, model_type, problem_type, metrics, params, feature_columns, target_column):
        """Log a new experiment run"""
        experiment_id = str(uuid.uuid4())[:8]
        
        experiment = {
            "id": experiment_id,
            "model_id": model_id,
            "dataset_name": dataset_name,
            "model_type": model_type,
            "problem_type": problem_type,
            "metrics": metrics,
            "params": params,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "created_at": datetime.now().isoformat(),
            "tags": [],
            "notes": ""
        }
        
        self.experiments[experiment_id] = experiment
        return experiment
    
    def get_all_experiments(self):
        """Return all experiments sorted by creation time"""
        exps = list(self.experiments.values())
        exps.sort(key=lambda x: x["created_at"], reverse=True)
        return exps
    
    def get_experiment(self, experiment_id):
        """Get a specific experiment"""
        return self.experiments.get(experiment_id)
    
    def delete_experiment(self, experiment_id):
        """Delete an experiment"""
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            return True
        return False
    
    def add_tag(self, experiment_id, tag):
        """Add a tag to an experiment"""
        if experiment_id in self.experiments:
            if tag not in self.experiments[experiment_id]["tags"]:
                self.experiments[experiment_id]["tags"].append(tag)
            return True
        return False
    
    def add_notes(self, experiment_id, notes):
        """Add notes to an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["notes"] = notes
            return True
        return False
