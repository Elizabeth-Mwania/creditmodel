import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, validation_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, log_loss
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from typing import Dict, List, Tuple, Optional, Any
import joblib
import pickle
import warnings
from loguru import logger
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class CreditScoringModelEngine:
    """
    This is a model fitting engine for credit scoring with multiple algorithms,
    hyperparameter tuning, and advanced evaluation metrics.
    """
    
    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        """
        Initialize the model engine
        
        Parameters:
        - random_state: Random seed for reproducibility
        - cv_folds: Number of cross-validation folds
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models = {}
        self.best_models = {}
        self.model_results = {}
        self.scalers = {}
        self.is_fitted = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize model configurations"""
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': self.random_state,
                    'max_iter': 1000
                },
                'param_grid': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None]
                },
                'scoring': 'roc_auc'
            },
            
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                },
                'scoring': 'roc_auc'
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'random_state': self.random_state
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'scoring': 'roc_auc'
            },
            
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'random_state': self.random_state
                },
                'param_grid': {
                    'max_depth': [3, 5, 7, 10, 15, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'criterion': ['gini', 'entropy'],
                    'class_weight': ['balanced', None]
                },
                'scoring': 'roc_auc'
            },
            
            'svm': {
                'model': SVC,
                'params': {
                    'random_state': self.random_state,
                    'probability': True
                },
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None]
                },
                'scoring': 'roc_auc'
            },
            
            'naive_bayes': {
                'model': GaussianNB,
                'params': {},
                'param_grid': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                },
                'scoring': 'roc_auc'
            },
            
            'knn': {
                'model': KNeighborsClassifier,
                'params': {},
                'param_grid': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'scoring': 'roc_auc'
            }
        }
    
    def apply_sampling_strategy(self, X: np.array, y: np.array, 
                               strategy: str = 'smote') -> Tuple[np.array, np.array]:
        """
        Apply various sampling strategies to handle class imbalance
        
        Parameters:
        - X: Feature matrix
        - y: Target vector
        - strategy: Sampling strategy ('smote', 'adasyn', 'borderline', 'tomek', 'smoteenn', 'smotetomek', 'undersample')
        
        Returns:
        - X_resampled, y_resampled: Resampled data
        """
        try:
            if strategy == 'smote':
                sampler = SMOTE(random_state=self.random_state)
            elif strategy == 'adasyn':
                sampler = ADASYN(random_state=self.random_state)
            elif strategy == 'borderline':
                sampler = BorderlineSMOTE(random_state=self.random_state)
            elif strategy == 'tomek':
                sampler = TomekLinks()
            elif strategy == 'smoteenn':
                sampler = SMOTEENN(random_state=self.random_state)
            elif strategy == 'smotetomek':
                sampler = SMOTETomek(random_state=self.random_state)
            elif strategy == 'undersample':
                sampler = RandomUnderSampler(random_state=self.random_state)
            else:
                logger.warning(f"Unknown sampling strategy: {strategy}. Returning original data.")
                return X, y
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            logger.info(f"Applied {strategy}: Original {X.shape} -> Resampled {X_resampled.shape}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Error applying sampling strategy {strategy}: {e}")
            return X, y
    
    def apply_scaling(self, X_train: np.array, X_test: np.array = None, 
                     method: str = 'standard') -> Tuple[np.array, np.array, Any]:
        """
        Apply feature scaling
        
        Parameters:
        - X_train: Training features
        - X_test: Test features (optional)
        - method: Scaling method ('standard', 'minmax', 'robust')
        
        Returns:
        - X_train_scaled, X_test_scaled, scaler
        """
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {method}. Using standard scaling.")
                scaler = StandardScaler()
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test) if X_test is not None else None
            
            return X_train_scaled, X_test_scaled, scaler
            
        except Exception as e:
            logger.error(f"Error applying scaling: {e}")
            return X_train, X_test, None
    
    def train_single_model(self, model_name: str, X_train: np.array, y_train: np.array,
                          hyperparameter_tuning: bool = True, 
                          tuning_method: str = 'grid',
                          scoring: str = 'roc_auc') -> Dict:
        """
        Train a single model with optional hyperparameter tuning
        
        Parameters:
        - model_name: Name of the model to train
        - X_train: Training features
        - y_train: Training targets
        - hyperparameter_tuning: Whether to perform hyperparameter tuning
        - tuning_method: 'grid' or 'random' search
        - scoring: Scoring metric for tuning
        
        Returns:
        - Dictionary with model results
        """
        try:
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            config = self.model_configs[model_name]
            model_class = config['model']
            base_params = config['params']
            param_grid = config['param_grid']
            
            logger.info(f"Training {model_name}...")
            
            if hyperparameter_tuning and param_grid:
                # Create cross-validation strategy
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                   random_state=self.random_state)
                
                # Initialize base model
                base_model = model_class(**base_params)
                
                # Perform hyperparameter tuning
                if tuning_method == 'grid':
                    search = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=cv,
                        n_jobs=-1,
                        verbose=0
                    )
                else:  # random search
                    search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=param_grid,
                        n_iter=50,
                        scoring=scoring,
                        cv=cv,
                        n_jobs=-1,
                        random_state=self.random_state,
                        verbose=0
                    )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_score = search.best_score_
                
                logger.info(f"{model_name} - Best CV {scoring}: {best_score:.4f}")
                logger.info(f"{model_name} - Best params: {best_params}")
                
            else:
                # Train with default parameters
                best_model = model_class(**base_params)
                best_model.fit(X_train, y_train)
                best_params = base_params
                
                # Calculate cross-validation score
                cv_scores = cross_val_score(best_model, X_train, y_train, 
                                          cv=self.cv_folds, scoring=scoring)
                best_score = cv_scores.mean()
                
                logger.info(f"{model_name} - CV {scoring}: {best_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return {
                'model': best_model,
                'best_params': best_params,
                'cv_score': best_score,
                'model_name': model_name
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {
                'model': None,
                'best_params': {},
                'cv_score': 0.0,
                'model_name': model_name,
                'error': str(e)
            }
    
    def train_multiple_models(self, X_train: np.array, y_train: np.array,
                            models_to_train: List[str] = None,
                            sampling_strategy: str = None,
                            scaling_method: str = None,
                            hyperparameter_tuning: bool = True) -> Dict:
        """
        Train multiple models and compare their performance
        
        Parameters:
        - X_train: Training features
        - y_train: Training targets
        - models_to_train: List of model names to train
        - sampling_strategy: Sampling strategy for imbalanced data
        - scaling_method: Feature scaling method
        - hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
        - Dictionary with all model results
        """
        try:
            if models_to_train is None:
                models_to_train = list(self.model_configs.keys())
            
            logger.info(f"Training {len(models_to_train)} models: {models_to_train}")
            
            # Apply sampling strategy if specified
            if sampling_strategy:
                X_train_processed, y_train_processed = self.apply_sampling_strategy(
                    X_train, y_train, sampling_strategy
                )
            else:
                X_train_processed, y_train_processed = X_train, y_train
            
            # Apply scaling if specified
            if scaling_method:
                X_train_scaled, _, scaler = self.apply_scaling(
                    X_train_processed, method=scaling_method
                )
                self.scalers['scaler'] = scaler
            else:
                X_train_scaled = X_train_processed
            
            results = {}
            
            for model_name in models_to_train:
                result = self.train_single_model(
                    model_name=model_name,
                    X_train=X_train_scaled,
                    y_train=y_train_processed,
                    hyperparameter_tuning=hyperparameter_tuning
                )
                
                results[model_name] = result
                
                if result['model'] is not None:
                    self.models[model_name] = result['model']
            
            self.model_results = results
            self.is_fitted = True
            
            logger.info("Model training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error training multiple models: {e}")
            raise
    
    def evaluate_models(self, X_test: np.array, y_test: np.array,
                       scaling_method: str = None) -> pd.DataFrame:
        """
        Evaluate all trained models on test data
        
        Parameters:
        - X_test: Test features
        - y_test: Test targets
        - scaling_method: Feature scaling method (should match training)
        
        Returns:
        - DataFrame with evaluation metrics for all models
        """
        try:
            if not self.is_fitted:
                raise ValueError("Models must be trained before evaluation")
            
            # Apply scaling if used during training
            if scaling_method and 'scaler' in self.scalers:
                X_test_scaled = self.scalers['scaler'].transform(X_test)
            else:
                X_test_scaled = X_test
            
            evaluation_results = []
            
            for model_name, model in self.models.items():
                try:
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    # Calculate metrics
                    metrics = {
                        'Model': model_name,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1_Score': f1_score(y_test, y_pred),
                        'ROC_AUC': roc_auc_score(y_test, y_prob),
                        'PR_AUC': average_precision_score(y_test, y_prob),
                        'Log_Loss': log_loss(y_test, y_prob) if len(np.unique(y_prob)) > 1 else np.nan
                    }
                    
                    # Add cross-validation score from training
                    if model_name in self.model_results:
                        metrics['CV_Score'] = self.model_results[model_name]['cv_score']
                    
                    evaluation_results.append(metrics)
                    
                    logger.info(f"{model_name} - Test ROC AUC: {metrics['ROC_AUC']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    continue
            
            results_df = pd.DataFrame(evaluation_results)
            results_df = results_df.sort_values('ROC_AUC', ascending=False)
            
            logger.info("Model evaluation completed!")
            return results_df
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            raise
    
    def get_best_model(self, metric: str = 'ROC_AUC') -> Tuple[str, Any]:
        """
        Get the best model based on specified metric
        
        Parameters:
        - metric: Metric to use for selection
        
        Returns:
        - Tuple of (model_name, model_object)
        """
        if not self.model_results:
            raise ValueError("No models have been trained yet")
        
        if metric == 'CV_Score':
            best_model_name = max(self.model_results.keys(), 
                                key=lambda x: self.model_results[x]['cv_score'])
        else:
            # This would need evaluation results - for now use CV score
            best_model_name = max(self.model_results.keys(), 
                                key=lambda x: self.model_results[x]['cv_score'])
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, output_folder: str):
        """
        Save all trained models and results
        
        Parameters:
        - output_folder: Directory to save models
        """
        try:
            os.makedirs(output_folder, exist_ok=True)
            
            # Save individual models
            for model_name, model in self.models.items():
                model_path = os.path.join(output_folder, f"{model_name}_model.pkl")
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")
            
            # Save scalers if used
            if self.scalers:
                scaler_path = os.path.join(output_folder, "scalers.pkl")
                joblib.dump(self.scalers, scaler_path)
            
            # Save model results
            results_path = os.path.join(output_folder, "model_training_results.json")
            serializable_results = {}
            for model_name, result in self.model_results.items():
                serializable_results[model_name] = {
                    'best_params': result['best_params'],
                    'cv_score': float(result['cv_score']),
                    'model_name': result['model_name']
                }
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Model results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, input_folder: str):
        """
        Load previously trained models
        
        Parameters:
        - input_folder: Directory containing saved models
        """
        try:
            # Load individual models
            for file in os.listdir(input_folder):
                if file.endswith('_model.pkl'):
                    model_name = file.replace('_model.pkl', '')
                    model_path = os.path.join(input_folder, file)
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} from {model_path}")
            
            # Load scalers if exist
            scaler_path = os.path.join(input_folder, "scalers.pkl")
            if os.path.exists(scaler_path):
                self.scalers = joblib.load(scaler_path)
            
            # Load results if exist
            results_path = os.path.join(input_folder, "model_training_results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.model_results = json.load(f)
            
            self.is_fitted = True
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise


class CreditScoringModelEvaluator:
    """
    Model evaluation with credit scoring specific metrics
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def calculate_gini_coefficient(self, y_true: np.array, y_prob: np.array) -> float:
        """Calculate Gini coefficient"""
        try:
            auc = roc_auc_score(y_true, y_prob)
            gini = 2 * auc - 1
            return gini
        except:
            return 0.0
    
    def calculate_ks_statistic(self, y_true: np.array, y_prob: np.array) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        try:
            # Separate good and bad customers
            good_scores = y_prob[y_true == 0]
            bad_scores = y_prob[y_true == 1]
            
            # Calculate KS statistic
            from scipy import stats
            ks_stat, _ = stats.ks_2samp(good_scores, bad_scores)
            return ks_stat
        except:
            return 0.0
    
    def create_evaluation_plots(self, y_true: np.array, y_prob: np.array, 
                               model_name: str, output_folder: str):
        """Create evaluation plots"""
        try:
            os.makedirs(output_folder, exist_ok=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
            axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title(f'{model_name} - ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            axes[0, 1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title(f'{model_name} - Precision-Recall Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Score Distribution
            good_scores = y_prob[y_true == 0]
            bad_scores = y_prob[y_true == 1]
            axes[1, 0].hist(good_scores, bins=50, alpha=0.7, label='Good', density=True)
            axes[1, 0].hist(bad_scores, bins=50, alpha=0.7, label='Bad', density=True)
            axes[1, 0].set_xlabel('Predicted Probability')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title(f'{model_name} - Score Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            
            # Confusion Matrix
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title(f'{model_name} - Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
            
            plt.tight_layout()
            plot_path = os.path.join(output_folder, f"{model_name}_evaluation_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Evaluation plots saved for {model_name}")
            
        except Exception as e:
            logger.error(f"Error creating evaluation plots for {model_name}: {e}")
    
    def model_evaluation(self, models_dict: Dict, X_test: np.array, 
                                     y_test: np.array, output_folder: str,
                                     scaler=None) -> pd.DataFrame:
        """
        Evaluate all models
        
        Parameters:
        - models_dict: Dictionary of trained models
        - X_test: Test features
        - y_test: Test targets
        - output_folder: Directory to save results
        - scaler: Optional scaler for feature scaling
        
        Returns:
        - DataFrame with detailed evaluation metrics
        """
        try:
            os.makedirs(output_folder, exist_ok=True)
            
            # Apply scaling if provided
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            evaluation_results = []
            
            for model_name, model in models_dict.items():
                try:
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_prob = y_pred
                    
                    # Calculate metrics
                    metrics = {
                        'Model': model_name,
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1_Score': f1_score(y_test, y_pred),
                        'ROC_AUC': roc_auc_score(y_test, y_prob),
                        'PR_AUC': average_precision_score(y_test, y_prob),
                        'Gini': self.calculate_gini_coefficient(y_test, y_prob),
                        'KS_Statistic': self.calculate_ks_statistic(y_test, y_prob),
                        'Log_Loss': log_loss(y_test, y_prob) if len(np.unique(y_prob)) > 1 else np.nan
                    }
                    
                    evaluation_results.append(metrics)
                    
                    # Create evaluation plots
                    self.create_evaluation_plots(y_test, y_prob, model_name, output_folder)
                    
                    logger.info(f"Evaluated {model_name} - ROC AUC: {metrics['ROC_AUC']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    continue
            
            # Create results DataFrame
            results_df = pd.DataFrame(evaluation_results)
            results_df = results_df.sort_values('ROC_AUC', ascending=False)
            
            # Save results to CSV
            results_path = os.path.join(output_folder, "model_evaluation_results.csv")
            results_df.to_csv(results_path, index=False)
            
            # Create comparison plot
            self._create_model_comparison_plot(results_df, output_folder)
            
            logger.info("Model evaluation completed!")
            return results_df
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise
    
    def _create_model_comparison_plot(self, results_df: pd.DataFrame, output_folder: str):
        """Create model comparison visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # ROC AUC comparison
            axes[0, 0].barh(results_df['Model'], results_df['ROC_AUC'])
            axes[0, 0].set_xlabel('ROC AUC')
            axes[0, 0].set_title('Model Comparison - ROC AUC')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Precision vs Recall
            axes[0, 1].scatter(results_df['Recall'], results_df['Precision'])
            for i, model in enumerate(results_df['Model']):
                axes[0, 1].annotate(model, (results_df.iloc[i]['Recall'], results_df.iloc[i]['Precision']))
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision vs Recall')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Gini coefficient comparison
            axes[1, 0].barh(results_df['Model'], results_df['Gini'])
            axes[1, 0].set_xlabel('Gini Coefficient')
            axes[1, 0].set_title('Model Comparison - Gini Coefficient')
            axes[1, 0].grid(True, alpha=0.3)
            
            # KS Statistic comparison
            axes[1, 1].barh(results_df['Model'], results_df['KS_Statistic'])
            axes[1, 1].set_xlabel('KS Statistic')
            axes[1, 1].set_title('Model Comparison - KS Statistic')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            comparison_path = os.path.join(output_folder, "model_comparison_plots.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Model comparison plots saved")
            
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {e}")