"""
Configuration file for credit scoring model fitting and evaluation
"""

# Model training configuration
MODEL_TRAINING_CONFIG = {
    "random_state": 42,
    "cv_folds": 5,
    "test_size": 0.2,
    "scoring_metric": "roc_auc",
    "hyperparameter_tuning": True,
    "tuning_method": "grid",  
    "n_jobs": -1
}

# Sampling strategies for imbalanced data
SAMPLING_STRATEGIES = {
    "smote": {
        "description": "Synthetic Minority Oversampling Technique",
        "recommended_for": "Most cases with moderate imbalance"
    },
    "adasyn": {
        "description": "Adaptive Synthetic Sampling",
        "recommended_for": "Complex decision boundaries"
    },
    "borderline": {
        "description": "Borderline SMOTE",
        "recommended_for": "Focus on difficult examples"
    },
    "smoteenn": {
        "description": "SMOTE + Edited Nearest Neighbours",
        "recommended_for": "Clean oversampled data"
    },
    "smotetomek": {
        "description": "SMOTE + Tomek Links",
        "recommended_for": "Remove noisy examples"
    },
    "undersample": {
        "description": "Random Undersampling",
        "recommended_for": "Large datasets with extreme imbalance"
    }
}

# Feature scaling methods
SCALING_METHODS = {
    "standard": {
        "description": "StandardScaler - zero mean, unit variance",
        "recommended_for": "Most algorithms, especially SVM, KNN"
    },
    "minmax": {
        "description": "MinMaxScaler - scale to [0,1] range",
        "recommended_for": "Neural networks, algorithms sensitive to outliers"
    },
    "robust": {
        "description": "RobustScaler - uses median and IQR",
        "recommended_for": "Data with outliers"
    }
}

# Default model configurations
DEFAULT_MODEL_PARAMS = {
    'logistic_regression': {
        'solver': 'liblinear',
        'penalty': 'l1',
        'C': 1.0,
        'class_weight': 'balanced',
        'max_iter': 1000,
        'random_state': 42
    },
    
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 5,
        'subsample': 0.9,
        'random_state': 42
    },
    
    'decision_tree': {
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'criterion': 'gini',
        'class_weight': 'balanced',
        'random_state': 42
    },
    
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42
    },
    
    'naive_bayes': {
        'var_smoothing': 1e-9
    },
    
    'knn': {
        'n_neighbors': 5,
        'weights': 'uniform',
        'metric': 'euclidean'
    }
}

# Hyperparameter grids for tuning
HYPERPARAMETER_GRIDS = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None]
    },
    
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    },
    
    'decision_tree': {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    },
    
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'class_weight': ['balanced', None]
    },
    
    'naive_bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },
    
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
}

# Credit scoring specific evaluation metrics
CREDIT_SCORING_METRICS = {
    'primary_metrics': ['ROC_AUC', 'Gini', 'KS_Statistic'],
    'secondary_metrics': ['Precision', 'Recall', 'F1_Score', 'PR_AUC'],
    'business_metrics': ['Accuracy', 'Log_Loss']
}

# Model selection preferences for credit scoring
MODEL_PREFERENCES = {
    'interpretability_ranking': [
        'logistic_regression',  # Most interpretable
        'decision_tree',
        'naive_bayes',
        'knn',
        'svm',
        'gradient_boosting',
        'random_forest'        # Least interpretable
    ],
    
    'performance_focus': [
        'gradient_boosting',   # Often best performance
        'random_forest',
        'logistic_regression',
        'svm',
        'decision_tree',
        'knn',
        'naive_bayes'
    ],
    
    'speed_ranking': [
        'naive_bayes',         # Fastest
        'logistic_regression',
        'knn',
        'decision_tree',
        'svm',
        'gradient_boosting',
        'random_forest'        # Slowest
    ]
}

# Evaluation thresholds
PERFORMANCE_THRESHOLDS = {
    'excellent': {
        'ROC_AUC': 0.85,
        'Gini': 0.70,
        'KS_Statistic': 0.40
    },
    'good': {
        'ROC_AUC': 0.75,
        'Gini': 0.50,
        'KS_Statistic': 0.30
    },
    'acceptable': {
        'ROC_AUC': 0.65,
        'Gini': 0.30,
        'KS_Statistic': 0.20
    },
    'poor': {
        'ROC_AUC': 0.55,
        'Gini': 0.10,
        'KS_Statistic': 0.10
    }
}

# Output file configurations
OUTPUT_CONFIG = {
    "model_files": {
        "model_results": "model_training_results.json",
        "evaluation_results": "model_evaluation_results.csv",
        "best_model": "best_model.pkl",
        "all_models": "all_models.pkl",
        "scalers": "scalers.pkl"
    },
    
    "plot_files": {
        "model_comparison": "model_comparison_plots.png",
        "feature_importance": "feature_importance_plots.png",
        "evaluation_plots": "{model_name}_evaluation_plots.png",
        "performance_summary": "performance_summary.png"
    },
    
    "report_files": {
        "training_report": "model_training_report.txt",
        "evaluation_report": "model_evaluation_report.txt",
        "model_summary": "model_summary.csv",
        "recommendations": "model_recommendations.txt"
    }
}

# Business rules for model selection
BUSINESS_RULES = {
    "regulatory_requirements": {
        "require_interpretability": True,
        "max_complexity_score": 0.7,  # Scale of 0 (simple) to 1 (complex)
        "documentation_required": True,
        "audit_trail_required": True
    },
    
    "performance_requirements": {
        "min_auc": 0.65,
        "min_gini": 0.30,
        "min_ks": 0.20,
        "max_false_positive_rate": 0.20,
        "max_false_negative_rate": 0.30
    },
    
    "operational_requirements": {
        "max_training_time_minutes": 60,
        "max_prediction_time_ms": 100,
        "max_model_size_mb": 100,
        "deployment_environment": "production"
    }
}

# Feature selection configuration
FEATURE_SELECTION_CONFIG = {
    "methods": {
        "statistical": {
            "chi2_threshold": 0.05,
            "mutual_info_threshold": 0.1
        },
        "model_based": {
            "use_feature_importance": True,
            "importance_threshold": 0.01,
            "recursive_elimination": False
        },
        "correlation": {
            "correlation_threshold": 0.95,
            "vif_threshold": 10
        }
    },
    
    "selection_criteria": {
        "max_features": 20,
        "min_features": 5,
        "stability_threshold": 0.8
    }
}

def get_model_config(model_name: str) -> dict:
    """Get default configuration for a specific model"""
    return DEFAULT_MODEL_PARAMS.get(model_name, {})

def get_hyperparameter_grid(model_name: str) -> dict:
    """Get hyperparameter grid for a specific model"""
    return HYPERPARAMETER_GRIDS.get(model_name, {})

def get_recommended_models_for_credit_scoring() -> list:
    """Get list of recommended models for credit scoring"""
    return [
        'logistic_regression',  # High interpretability, good baseline
        'random_forest',        # Good performance, feature importance
        'gradient_boosting',    # Often best performance
        'decision_tree'         # High interpretability
    ]

def validate_model_config():
    """Validate model configuration"""
    errors = []
    
    # Check that all models have both default params and hyperparameter grids
    models_with_defaults = set(DEFAULT_MODEL_PARAMS.keys())
    models_with_grids = set(HYPERPARAMETER_GRIDS.keys())
    
    missing_defaults = models_with_grids - models_with_defaults
    missing_grids = models_with_defaults - models_with_grids
    
    if missing_defaults:
        errors.append(f"Models missing default parameters: {missing_defaults}")
    if missing_grids:
        errors.append(f"Models missing hyperparameter grids: {missing_grids}")
    
    # Validate performance thresholds
    for level, thresholds in PERFORMANCE_THRESHOLDS.items():
        for metric, value in thresholds.items():
            if not 0 <= value <= 1:
                errors.append(f"Invalid threshold for {level}.{metric}: {value}")
    
    if errors:
        raise ValueError("Configuration validation failed: " + "; ".join(errors))
    
    return True

def print_model_config_summary():
    """Print summary of model configuration"""
    print("=== MODEL CONFIGURATION SUMMARY ===")
    print(f"Available models: {len(DEFAULT_MODEL_PARAMS)}")
    
    for model_name in DEFAULT_MODEL_PARAMS.keys():
        print(f"- {model_name}")
    
    print(f"\nSampling strategies: {len(SAMPLING_STRATEGIES)}")
    print(f"Scaling methods: {len(SCALING_METHODS)}")
    print(f"Cross-validation folds: {MODEL_TRAINING_CONFIG['cv_folds']}")
    print(f"Primary scoring metric: {MODEL_TRAINING_CONFIG['scoring_metric']}")
    
    recommended = get_recommended_models_for_credit_scoring()
    print(f"\nRecommended models for credit scoring: {recommended}")

if __name__ == "__main__":
    validate_model_config()
    print_model_config_summary()
    print("Model configuration validation passed!")