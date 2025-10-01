import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from loguru import logger
import warnings
from datetime import datetime
import joblib
import json

# Import our custom modules
try:
    from model_fitting_engine import CreditScoringModelEngine, CreditScoringModelEvaluator
    from model_config import (
        MODEL_TRAINING_CONFIG,
        DEFAULT_MODEL_PARAMS,
        get_recommended_models_for_credit_scoring,
        validate_model_config,
        print_model_config_summary,
        OUTPUT_CONFIG,
        PERFORMANCE_THRESHOLDS
    )
    # Import scorecard utilities
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03.binning_process'))
    from scorecard_utils import CreditScorecard, generate_scorecard_report
    IMPORTS_AVAILABLE = True
    print("All model fitting modules imported successfully!")

except ImportError as e:
    print(f"Model fitting modules not found: {e}")
    IMPORTS_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set_style({'axes.grid': False})

def set_project_root():
    """Set the working directory to the project root"""
    try:
        current_file_path = os.path.abspath(__file__)
        project_root = current_file_path

        while not os.path.exists(os.path.join(project_root, ".git")):
            project_root = os.path.dirname(project_root)
            if project_root == os.path.dirname(project_root):
                project_root = os.getcwd()
                break

        os.chdir(project_root)
        logger.info(f"Current working directory set to: {os.getcwd()}")
        return project_root
    except Exception as e:
        logger.error(f"Failed to set project root: {e}")
        return os.getcwd()

def create_directory_if_not_exists(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory created: {path}")

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def prepare_features_target(train_df, test_df, selected_features):
    """
    Prepare features and target variables from the datasets
    
    Parameters:
    - train_df: Training DataFrame
    - test_df: Test DataFrame  
    - selected_features: List of selected feature names
    
    Returns:
    - X_train, y_train, X_test, y_test
    """
    try:
        # Verify all selected features exist in the data
        missing_features = [f for f in selected_features if f not in train_df.columns]
        if missing_features:
            logger.warning(f"Missing features in training data: {missing_features}")
            selected_features = [f for f in selected_features if f in train_df.columns]
        
        # Prepare training data
        X_train = train_df[selected_features].copy()
        y_train = train_df['TARGET'].copy()
        
        # Prepare test data
        X_test = test_df[selected_features].copy()
        y_test = test_df['TARGET'].copy()
        
        # Check for missing values
        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
        
        if train_missing > 0:
            logger.warning(f"Found {train_missing} missing values in training features")
            X_train = X_train.fillna(X_train.median())
        
        if test_missing > 0:
            logger.warning(f"Found {test_missing} missing values in test features")
            X_test = X_test.fillna(X_train.median())  # Use training median for test
        
        logger.info(f"Prepared features: {len(selected_features)} features")
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train.values, y_train.values, X_test.values, y_test.values, selected_features
        
    except Exception as e:
        logger.error(f"Error preparing features and target: {e}")
        raise

def create_model_training_report(model_engine, results_df, output_folder):
    """Create comprehensive training report"""
    try:
        report_path = os.path.join(output_folder, OUTPUT_CONFIG["report_files"]["training_report"])
        
        with open(report_path, 'w') as f:
            f.write("CREDIT SCORING MODEL TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training configuration
            f.write("TRAINING CONFIGURATION:\n")
            f.write(f"- Random State: {MODEL_TRAINING_CONFIG['random_state']}\n")
            f.write(f"- Cross-Validation Folds: {MODEL_TRAINING_CONFIG['cv_folds']}\n")
            f.write(f"- Hyperparameter Tuning: {MODEL_TRAINING_CONFIG['hyperparameter_tuning']}\n")
            f.write(f"- Scoring Metric: {MODEL_TRAINING_CONFIG['scoring_metric']}\n\n")
            
            # Models trained
            f.write("MODELS TRAINED:\n")
            for i, row in results_df.iterrows():
                f.write(f"- {row['Model']}: ROC AUC = {row['ROC_AUC']:.4f}\n")
            f.write("\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY:\n")
            best_model = results_df.iloc[0]
            f.write(f"- Best Model: {best_model['Model']}\n")
            f.write(f"- Best ROC AUC: {best_model['ROC_AUC']:.4f}\n")
            f.write(f"- Best Gini: {best_model['Gini']:.4f}\n")
            f.write(f"- Best KS Statistic: {best_model['KS_Statistic']:.4f}\n\n")
            
            # Performance assessment
            f.write("PERFORMANCE ASSESSMENT:\n")
            auc_score = best_model['ROC_AUC']
            if auc_score >= PERFORMANCE_THRESHOLDS['excellent']['ROC_AUC']:
                assessment = "Excellent"
            elif auc_score >= PERFORMANCE_THRESHOLDS['good']['ROC_AUC']:
                assessment = "Good"
            elif auc_score >= PERFORMANCE_THRESHOLDS['acceptable']['ROC_AUC']:
                assessment = "Acceptable"
            else:
                assessment = "Poor - Consider feature engineering or more data"
            
            f.write(f"- Overall Assessment: {assessment}\n")
            f.write(f"- Recommendation: {'Deploy with confidence' if assessment in ['Excellent', 'Good'] else 'Review and improve before deployment'}\n")
            
        logger.info(f"Training report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error creating training report: {e}")

def create_feature_importance_analysis(model_engine, feature_names, output_folder):
    """Create feature importance analysis for interpretable models"""
    try:
        feature_importance_data = []
        
        for model_name, model in model_engine.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importances = model.feature_importances_
                    for i, importance in enumerate(importances):
                        feature_importance_data.append({
                            'Model': model_name,
                            'Feature': feature_names[i],
                            'Importance': importance
                        })
                elif hasattr(model, 'coef_'):
                    # Linear models
                    coefficients = np.abs(model.coef_[0])
                    for i, coef in enumerate(coefficients):
                        feature_importance_data.append({
                            'Model': model_name,
                            'Feature': feature_names[i],
                            'Importance': coef
                        })
            except Exception as e:
                logger.warning(f"Could not extract feature importance for {model_name}: {e}")
                continue
        
        if feature_importance_data:
            importance_df = pd.DataFrame(feature_importance_data)
            
            # Save to CSV
            importance_path = os.path.join(output_folder, "feature_importance_analysis.csv")
            importance_df.to_csv(importance_path, index=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Get top features across all models
            avg_importance = importance_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
            top_features = avg_importance.head(15)
            
            plt.barh(range(len(top_features)), top_features.values)
            plt.yticks(range(len(top_features)), top_features.index)
            plt.xlabel('Average Importance')
            plt.title('Top 15 Features by Average Importance Across Models')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_folder, OUTPUT_CONFIG["plot_files"]["feature_importance"])
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance analysis saved")
            
    except Exception as e:
        logger.error(f"Error creating feature importance analysis: {e}")

def main():
    """Main model fitting and evaluation pipeline"""
    
    # Set up logging
    log_file = "model_fitting.log"
    logger.add(log_file, rotation="500 MB", level="INFO")
    
    logger.info("="*80)
    logger.info("STARTING CREDIT SCORING MODEL FITTING PIPELINE")
    logger.info("="*80)
    
    # Check if custom modules are available
    if not IMPORTS_AVAILABLE:
        logger.error("Required modules not found. Please ensure model_fitting_engine.py and model_config.py exist.")
        return
    
    try:
        # Validate configuration
        validate_model_config()
        print_model_config_summary()
        
        # Set project root
        project_root = set_project_root()
        
        # Define paths
        current_dir = os.getcwd()
        input_dir = "03.binning_process\\outputs"
        output_dir = "04.model_fitting\\outputs"
        
        train_file = "03.train_data.csv"
        test_file = "03.test_data.csv"
        
        train_path = os.path.join(current_dir, input_dir, train_file)
        test_path = os.path.join(current_dir, input_dir, test_file)
        
        # Create output directory
        create_directory_if_not_exists(os.path.join(current_dir, output_dir))
        output_folder = os.path.join(current_dir, output_dir)
        
        # Load data
        logger.info("="*50)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*50)
        
        train_df = load_data(train_path)
        test_df = load_data(test_path)
        
        # Define selected features (from your binning process results)
        selected_features =[
            "days_since_last_txn",
            "txn_count_7d",
            "sum_amount_7d",
            "avg_amount_7d",
            "max_amount_7d",
            "min_amount_7d",
            "std_amount_7d",
            "cv_amount_7d",
            "mean_abs_z_7d",
            "max_abs_z_7d",
            "avg_txn_per_day_7d",
            "amount_min_max_ratio_7d",
            "avg_amount_1m",
            "max_amount_1m",
            "min_amount_1m",
            "std_amount_1m",
            "cv_amount_1m",
            "mean_abs_z_1m",
            "max_abs_z_1m",
            "active_days_1m",
            "amount_min_max_ratio_1m",
            "avg_amount_3m",
            "max_amount_3m",
            "min_amount_3m",
            "max_abs_z_3m",
            "txn_count_6m",
            "avg_amount_6m",
            "avg_txn_per_day_6m",
            "txn_count_overall",
            "dependency_ratio_7d",
            "dependency_ratio_1m"
        ]

        
        logger.info(f"Selected features: {len(selected_features)}")
        
        # Prepare features and target
        logger.info("="*50)
        logger.info("STEP 2: PREPARING FEATURES")
        logger.info("="*50)
        
        X_train, y_train, X_test, y_test, final_features = prepare_features_target(
            train_df, test_df, selected_features
        )
        
        # Check class distribution
        train_class_dist = np.bincount(y_train.astype(int)) / len(y_train)
        test_class_dist = np.bincount(y_test.astype(int)) / len(y_test)
        
        logger.info(f"Training set class distribution: {train_class_dist}")
        logger.info(f"Test set class distribution: {test_class_dist}")
        
        # Initialize model engine
        logger.info("="*50)
        logger.info("STEP 3: INITIALIZING MODEL ENGINE")
        logger.info("="*50)
        
        model_engine = CreditScoringModelEngine(
            random_state=MODEL_TRAINING_CONFIG['random_state'],
            cv_folds=MODEL_TRAINING_CONFIG['cv_folds']
        )
        
        # Get recommended models for credit scoring
        models_to_train = get_recommended_models_for_credit_scoring()
        logger.info(f"Models to train: {models_to_train}")
        
        # Train models
        logger.info("="*50)
        logger.info("STEP 4: TRAINING MODELS")
        logger.info("="*50)
        
        # Train multiple models with different configurations
        training_results = model_engine.train_multiple_models(
            X_train=X_train,
            y_train=y_train,
            models_to_train=models_to_train,
            sampling_strategy='smote',  # Handle class imbalance
            scaling_method='standard',   # Scale features
            hyperparameter_tuning=MODEL_TRAINING_CONFIG['hyperparameter_tuning']
        )
        
        logger.info(f"Successfully trained {len(training_results)} models")
        
        # Evaluate models
        logger.info("="*50)
        logger.info("STEP 5: EVALUATING MODELS")
        logger.info("="*50)
        
        # Initialize evaluator
        evaluator = CreditScoringModelEvaluator()
        
        # Perform comprehensive evaluation
        evaluation_results = evaluator.model_evaluation(
            models_dict=model_engine.models,
            X_test=X_test,
            y_test=y_test,
            output_folder=output_folder,
            scaler=model_engine.scalers.get('scaler')
        )
        
        logger.info("Model evaluation completed")
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        print(evaluation_results.to_string(index=False))

        # Create scorecard if logistic regression was trained
        scorecard_results = None
        if 'logistic_regression' in model_engine.models:
            logger.info("="*50)
            logger.info("STEP 5.5: CREATING SCORECARD")
            logger.info("="*50)

            try:
                # Create scorecard using the trained logistic regression
                lr_model = model_engine.models['logistic_regression']

                # Create scorecard with simplified WoE mappings
                scorecard = CreditScorecard(base_score=600, pdo=20, odds=20)

                # Calculate points manually using coefficients
                coef = lr_model.coef_[0]
                intercept = lr_model.intercept_[0]

                factor = 20 / np.log(2)  # PDO = 20
                offset = 600 - factor * np.log(20)  # Base score 600, odds 20:1

                scorecard_table = {'base_points': round(offset + factor * intercept)}

                for i, feature in enumerate(final_features):
                    # Simplified: assume 3 bins per feature with WoE -0.5, 0, 0.5
                    feature_coef = coef[i]
                    scorecard_table[feature] = {
                        'Low': round(factor * (feature_coef * -0.5)),
                        'Medium': round(factor * (feature_coef * 0.0)),
                        'High': round(factor * (feature_coef * 0.5))
                    }

                scorecard.scorecard_table = scorecard_table
                scorecard.is_fitted = True

                # Calculate scores for test data
                test_df_features = pd.DataFrame(X_test, columns=final_features)
                scores = scorecard.calculate_score(test_df_features, final_features)

                # Generate scorecard report
                scorecard_output_dir = os.path.join(output_folder, "scorecard")
                generate_scorecard_report(scorecard, scores, y_test, scorecard_output_dir)

                scorecard_results = {
                    'scorecard': scorecard,
                    'scores': scores,
                    'scorecard_dir': scorecard_output_dir
                }

                logger.info("Scorecard created successfully")
                print("\n" + "="*60)
                print("SCORECARD CREATED")
                print("="*60)
                print(f"Scorecard saved to: {scorecard_output_dir}")

            except Exception as e:
                logger.error(f"Error creating scorecard: {e}")

        # Save models and results
        logger.info("="*50)
        logger.info("STEP 6: SAVING MODELS AND RESULTS")
        logger.info("="*50)
        
        model_engine.save_models(output_folder)
        
        # Save best model separately
        best_model_name, best_model = model_engine.get_best_model('CV_Score')
        best_model_path = os.path.join(output_folder, OUTPUT_CONFIG["model_files"]["best_model"])
        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model ({best_model_name}) saved to: {best_model_path}")
        
        # Create comprehensive reports
        logger.info("="*50)
        logger.info("STEP 7: GENERATING REPORTS")
        logger.info("="*50)
        
        create_model_training_report(model_engine, evaluation_results, output_folder)
        create_feature_importance_analysis(model_engine, final_features, output_folder)
        
        # Create model summary
        model_summary = evaluation_results[['Model', 'ROC_AUC', 'Gini', 'KS_Statistic', 'Precision', 'Recall', 'F1_Score']]
        summary_path = os.path.join(output_folder, OUTPUT_CONFIG["report_files"]["model_summary"])
        model_summary.to_csv(summary_path, index=False)
        
        # Final summary
        logger.info("="*50)
        logger.info("STEP 8: FINAL SUMMARY")
        logger.info("="*50)
        
        best_result = evaluation_results.iloc[0]
        
        print("\n" + "="*80)
        print("MODEL FITTING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best Model: {best_result['Model']}")
        print(f"Best ROC AUC: {best_result['ROC_AUC']:.4f}")
        print(f"Best Gini Coefficient: {best_result['Gini']:.4f}")
        print(f"Best KS Statistic: {best_result['KS_Statistic']:.4f}")
        print(f"Models trained: {len(training_results)}")
        print(f"Features used: {len(final_features)}")
        print(f"Output directory: {output_folder}")
        
        # Performance assessment
        auc_score = best_result['ROC_AUC']
        if auc_score >= PERFORMANCE_THRESHOLDS['excellent']['ROC_AUC']:
            assessment = "EXCELLENT"
        elif auc_score >= PERFORMANCE_THRESHOLDS['good']['ROC_AUC']:
            assessment = "GOOD"
        elif auc_score >= PERFORMANCE_THRESHOLDS['acceptable']['ROC_AUC']:
            assessment = "ACCEPTABLE"
        else:
            assessment = "NEEDS IMPROVEMENT"
        
        print(f"\nPerformance Assessment: {assessment}")
        
        # Model recommendations
        print(f"\nModel Rankings by ROC AUC:")
        for i, row in evaluation_results.head(5).iterrows():
            print(f"{i+1}. {row['Model']}: {row['ROC_AUC']:.4f}")
        
        # Key output files
        print(f"\nKey Output Files:")
        print(f"- Best model: {OUTPUT_CONFIG['model_files']['best_model']}")
        print(f"- All models: Multiple *_model.pkl files")
        print(f"- Evaluation results: {OUTPUT_CONFIG['model_files']['evaluation_results']}")
        print(f"- Training report: {OUTPUT_CONFIG['report_files']['training_report']}")
        print(f"- Model summary: {OUTPUT_CONFIG['report_files']['model_summary']}")
        
        logger.info("Model fitting pipeline completed successfully!")
        
        # Return results for potential further use
        return {
            'best_model_name': best_model_name,
            'best_model': best_model,
            'evaluation_results': evaluation_results,
            'model_engine': model_engine,
            'output_folder': output_folder,
            'selected_features': final_features
        }
        
    except Exception as e:
        logger.error(f"Error in main model fitting pipeline: {e}")
        raise

if __name__ == "__main__":
    main()