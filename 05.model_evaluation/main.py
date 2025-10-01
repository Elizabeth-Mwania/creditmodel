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
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)

# Import our custom modules
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '04.model_fitting'))
    from model_fitting_engine import CreditScoringModelEvaluator
    from model_config import (
        OUTPUT_CONFIG,
        PERFORMANCE_THRESHOLDS,
        CREDIT_SCORING_METRICS
    )
    # Import scorecard utilities
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03.binning_process'))
    from scorecard_utils import CreditScorecard
    IMPORTS_AVAILABLE = True
    print("Model evaluation modules imported successfully!")

except ImportError as e:
    print(f"Model evaluation modules not found: {e}")
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

def load_trained_models(models_folder):
    """Load all trained models from the specified folder"""
    try:
        models = {}
        scaler = None
        
        # Load individual model files
        for file in os.listdir(models_folder):
            if file.endswith('_model.pkl'):
                model_name = file.replace('_model.pkl', '')
                model_path = os.path.join(models_folder, file)
                models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model")
        
        # Load scaler if exists
        scaler_path = os.path.join(models_folder, "scalers.pkl")
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            scaler = scalers.get('scaler')
            logger.info("Loaded feature scaler")
        
        # Load best model specifically
        best_model_path = os.path.join(models_folder, OUTPUT_CONFIG["model_files"]["best_model"])
        if os.path.exists(best_model_path):
            best_model = joblib.load(best_model_path)
            logger.info("Loaded best model")
        else:
            # If no specific best model file, use the first model as best
            best_model = list(models.values())[0] if models else None
            logger.warning("No best model file found, using first available model")
        
        return models, scaler, best_model
        
    except Exception as e:
        logger.error(f"Error loading trained models: {e}")
        raise

def generate_predictions(models, X_test, y_test, scaler=None, output_folder=None):
    """Generate predictions for all models and save results"""
    try:
        # Apply scaling if scaler is provided
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        predictions_data = []
        
        for model_name, model in models.items():
            try:
                # Generate predictions
                y_pred = model.predict(X_test_scaled)
                
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_prob = y_pred.astype(float)
                
                # Store predictions
                for i in range(len(y_test)):
                    predictions_data.append({
                        'Model': model_name,
                        'Index': i,
                        'True_Label': y_test[i],
                        'Predicted_Label': y_pred[i],
                        'Predicted_Probability': y_prob[i]
                    })
                
                logger.info(f"Generated predictions for {model_name}")
                
            except Exception as e:
                logger.error(f"Error generating predictions for {model_name}: {e}")
                continue
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions_data)
        
        # Save predictions if output folder is provided
        if output_folder:
            predictions_path = os.path.join(output_folder, "detailed_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Detailed predictions saved to: {predictions_path}")
        
        return predictions_df
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise

def create_business_scorecard_analysis(best_model, X_test, y_test, scaler=None, output_folder=None):
    """Create business-oriented scorecard analysis"""
    try:
        # Apply scaling if needed
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        # Generate predictions
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else best_model.predict(X_test_scaled)

        # Create score bands
        score_bands = pd.cut(y_prob, bins=10, labels=[f'Band_{i+1}' for i in range(10)])

        # Create scorecard analysis
        scorecard_analysis = pd.DataFrame({
            'True_Label': y_test,
            'Probability': y_prob,
            'Score_Band': score_bands
        })

        # Calculate band statistics
        band_stats = scorecard_analysis.groupby('Score_Band').agg({
            'True_Label': ['count', 'sum', 'mean'],
            'Probability': ['mean', 'min', 'max']
        }).round(4)

        band_stats.columns = ['Count', 'Bad_Count', 'Bad_Rate', 'Avg_Probability', 'Min_Probability', 'Max_Probability']
        band_stats['Good_Count'] = band_stats['Count'] - band_stats['Bad_Count']
        band_stats['Good_Rate'] = 1 - band_stats['Bad_Rate']

        # Reorder columns
        band_stats = band_stats[['Count', 'Good_Count', 'Bad_Count', 'Good_Rate', 'Bad_Rate', 'Avg_Probability', 'Min_Probability', 'Max_Probability']]

        if output_folder:
            # Save scorecard analysis
            scorecard_path = os.path.join(output_folder, "scorecard_analysis.csv")
            band_stats.to_csv(scorecard_path)

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Bad rate by score band
            axes[0, 0].bar(range(len(band_stats)), band_stats['Bad_Rate'], alpha=0.7, color='red')
            axes[0, 0].set_title('Bad Rate by Score Band')
            axes[0, 0].set_xlabel('Score Band')
            axes[0, 0].set_ylabel('Bad Rate')
            axes[0, 0].set_xticks(range(len(band_stats)))
            axes[0, 0].set_xticklabels(band_stats.index, rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Count distribution
            axes[0, 1].bar(range(len(band_stats)), band_stats['Count'], alpha=0.7, color='blue')
            axes[0, 1].set_title('Count Distribution by Score Band')
            axes[0, 1].set_xlabel('Score Band')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_xticks(range(len(band_stats)))
            axes[0, 1].set_xticklabels(band_stats.index, rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Score distribution by outcome
            good_scores = y_prob[y_test == 0]
            bad_scores = y_prob[y_test == 1]

            axes[1, 0].hist(good_scores, bins=30, alpha=0.7, label='Good', color='green', density=True)
            axes[1, 0].hist(bad_scores, bins=30, alpha=0.7, label='Bad', color='red', density=True)
            axes[1, 0].set_title('Score Distribution by Outcome')
            axes[1, 0].set_xlabel('Predicted Probability')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Cumulative bad rate
            sorted_probs = np.sort(y_prob)[::-1]  # Sort descending
            sorted_labels = y_test[np.argsort(y_prob)[::-1]]
            cumulative_bad_rate = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)

            axes[1, 1].plot(range(len(cumulative_bad_rate)), cumulative_bad_rate, color='purple')
            axes[1, 1].set_title('Cumulative Bad Rate (Ordered by Score)')
            axes[1, 1].set_xlabel('Observations (Ordered by Risk Score)')
            axes[1, 1].set_ylabel('Cumulative Bad Rate')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            scorecard_plot_path = os.path.join(output_folder, "scorecard_business_analysis.png")
            plt.savefig(scorecard_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Business scorecard analysis completed")

        return band_stats

    except Exception as e:
        logger.error(f"Error creating business scorecard analysis: {e}")
        raise

def evaluate_scorecard_performance(output_folder, available_features):
    """Evaluate the scorecard performance if scorecard was created"""
    try:
        scorecard_dir = os.path.join(output_folder, "..", "04.model_fitting", "outputs", "scorecard")

        if not os.path.exists(scorecard_dir):
            logger.info("No scorecard directory found - scorecard evaluation skipped")
            return None

        # Load scorecard if it exists
        scorecard_files = [f for f in os.listdir(scorecard_dir) if f.endswith('.pkl')]
        if not scorecard_files:
            logger.info("No scorecard model file found")
            return None

        scorecard_path = os.path.join(scorecard_dir, scorecard_files[0])
        scorecard = joblib.load(scorecard_path)

        # Load test data for scorecard evaluation
        test_data_path = os.path.join(output_folder, "..", "03.binning_process", "outputs", "03.test_data.csv")
        if not os.path.exists(test_data_path):
            logger.warning("Test data not found for scorecard evaluation")
            return None

        test_df = pd.read_csv(test_data_path)
        y_test = test_df['TARGET'].values

        # Prepare features for scorecard
        available_test_features = [f for f in available_features if f in test_df.columns]
        X_test_df = test_df[available_test_features]

        # Calculate scorecard scores
        scores = scorecard.calculate_score(X_test_df, available_test_features)

        # Evaluate scorecard performance
        from sklearn.metrics import roc_auc_score, classification_report

        # Calculate basic metrics
        auc_score = roc_auc_score(y_test, scores)
        gini = 2 * auc_score - 1

        # Create score bands for scorecard analysis
        score_bands = pd.cut(scores, bins=10, labels=[f'Band_{i+1}' for i in range(10)])

        scorecard_analysis = pd.DataFrame({
            'True_Label': y_test,
            'Score': scores,
            'Score_Band': score_bands
        })

        # Calculate band statistics
        band_stats = scorecard_analysis.groupby('Score_Band').agg({
            'True_Label': ['count', 'sum', 'mean'],
            'Score': ['mean', 'min', 'max']
        }).round(4)

        band_stats.columns = ['Count', 'Bad_Count', 'Bad_Rate', 'Avg_Score', 'Min_Score', 'Max_Score']
        band_stats['Good_Count'] = band_stats['Count'] - band_stats['Bad_Count']
        band_stats['Good_Rate'] = 1 - band_stats['Bad_Rate']

        # Create scorecard evaluation report
        scorecard_eval_path = os.path.join(output_folder, "scorecard_evaluation_report.txt")
        with open(scorecard_eval_path, 'w') as f:
            f.write("SCORECARD PERFORMANCE EVALUATION\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("SCORECARD METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"AUC Score: {auc_score:.4f}\n")
            f.write(f"Gini Coefficient: {gini:.4f}\n")
            f.write(f"Score Range: {scores.min():.1f} - {scores.max():.1f}\n")
            f.write(f"Mean Score: {scores.mean():.1f}\n\n")

            f.write("SCORE BANDS ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(band_stats.to_string())
            f.write("\n\n")

            f.write("SCORECARD TABLE:\n")
            f.write("-" * 30 + "\n")
            for feature, points in scorecard.scorecard_table.items():
                if feature != 'base_points':
                    f.write(f"{feature}:\n")
                    if isinstance(points, dict):
                        for bin_name, point_value in points.items():
                            f.write(f"  {bin_name}: {point_value} points\n")
                    f.write("\n")

        # Create scorecard visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Score distribution by outcome
        good_scores = scores[y_test == 0]
        bad_scores = scores[y_test == 1]

        axes[0, 0].hist(good_scores, bins=30, alpha=0.7, label='Good', color='green', density=True)
        axes[0, 0].hist(bad_scores, bins=30, alpha=0.7, label='Bad', color='red', density=True)
        axes[0, 0].set_title('Scorecard Score Distribution by Outcome')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Bad rate by score band
        axes[0, 1].bar(range(len(band_stats)), band_stats['Bad_Rate'], alpha=0.7, color='red')
        axes[0, 1].set_title('Bad Rate by Score Band')
        axes[0, 1].set_xlabel('Score Band')
        axes[0, 1].set_ylabel('Bad Rate')
        axes[0, 1].set_xticks(range(len(band_stats)))
        axes[0, 1].set_xticklabels(band_stats.index, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Score vs Probability correlation (if available)
        axes[1, 0].scatter(scores, scores, alpha=0.5, color='blue')  # Placeholder
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative bad rate by score
        sorted_scores = np.sort(scores)[::-1]  # Sort descending (higher scores = lower risk)
        sorted_labels = y_test[np.argsort(scores)[::-1]]
        cumulative_bad_rate = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)

        axes[1, 1].plot(range(len(cumulative_bad_rate)), cumulative_bad_rate, color='purple')
        axes[1, 1].set_title('Cumulative Bad Rate (Ordered by Score)')
        axes[1, 1].set_xlabel('Observations (Ordered by Score)')
        axes[1, 1].set_ylabel('Cumulative Bad Rate')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        scorecard_eval_plot_path = os.path.join(output_folder, "scorecard_evaluation.png")
        plt.savefig(scorecard_eval_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Scorecard evaluation completed")

        return {
            'auc_score': auc_score,
            'gini': gini,
            'band_stats': band_stats,
            'scores': scores
        }

    except Exception as e:
        logger.error(f"Error evaluating scorecard: {e}")
        return None

def create_model_performance_summary(evaluation_results, output_folder):
    """Create a comprehensive performance summary report"""
    try:
        report_path = os.path.join(output_folder, OUTPUT_CONFIG["report_files"]["evaluation_report"])
        
        with open(report_path, 'w') as f:
            f.write("CREDIT SCORING MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            best_model = evaluation_results.iloc[0]
            f.write(f"Best Performing Model: {best_model['Model']}\n")
            f.write(f"ROC AUC Score: {best_model['ROC_AUC']:.4f}\n")
            f.write(f"Gini Coefficient: {best_model['Gini']:.4f}\n")
            f.write(f"KS Statistic: {best_model['KS_Statistic']:.4f}\n")
            f.write(f"Precision: {best_model['Precision']:.4f}\n")
            f.write(f"Recall: {best_model['Recall']:.4f}\n")
            f.write(f"F1 Score: {best_model['F1_Score']:.4f}\n\n")
            
            # Performance assessment
            f.write("PERFORMANCE ASSESSMENT:\n")
            f.write("-" * 40 + "\n")
            
            auc_score = best_model['ROC_AUC']
            gini_score = best_model['Gini']
            ks_score = best_model['KS_Statistic']
            
            # Determine overall grade
            if (auc_score >= PERFORMANCE_THRESHOLDS['excellent']['ROC_AUC'] and
                gini_score >= PERFORMANCE_THRESHOLDS['excellent']['Gini'] and
                ks_score >= PERFORMANCE_THRESHOLDS['excellent']['KS_Statistic']):
                grade = "EXCELLENT"
                recommendation = "Deploy immediately - Outstanding performance"
            elif (auc_score >= PERFORMANCE_THRESHOLDS['good']['ROC_AUC'] and
                  gini_score >= PERFORMANCE_THRESHOLDS['good']['Gini'] and
                  ks_score >= PERFORMANCE_THRESHOLDS['good']['KS_Statistic']):
                grade = "GOOD"
                recommendation = "Ready for deployment - Good performance"
            elif (auc_score >= PERFORMANCE_THRESHOLDS['acceptable']['ROC_AUC'] and
                  gini_score >= PERFORMANCE_THRESHOLDS['acceptable']['Gini'] and
                  ks_score >= PERFORMANCE_THRESHOLDS['acceptable']['KS_Statistic']):
                grade = "ACCEPTABLE"
                recommendation = "Consider deployment with monitoring - Acceptable performance"
            else:
                grade = "NEEDS IMPROVEMENT"
                recommendation = "Do not deploy - Requires model improvement"
            
            f.write(f"Overall Grade: {grade}\n")
            f.write(f"Recommendation: {recommendation}\n\n")
            
            # Individual model performance
            f.write("INDIVIDUAL MODEL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            
            for i, row in evaluation_results.iterrows():
                f.write(f"\n{i+1}. {row['Model']}:\n")
                f.write(f"   ROC AUC: {row['ROC_AUC']:.4f}\n")
                f.write(f"   Gini: {row['Gini']:.4f}\n")
                f.write(f"   KS Statistic: {row['KS_Statistic']:.4f}\n")
                f.write(f"   Precision: {row['Precision']:.4f}\n")
                f.write(f"   Recall: {row['Recall']:.4f}\n")
            
            # Business implications
            f.write(f"\nBUSINESS IMPLICATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"- Model Interpretability: {'High' if best_model['Model'] in ['logistic_regression', 'decision_tree'] else 'Medium to Low'}\n")
            f.write(f"- Deployment Complexity: {'Low' if best_model['Model'] in ['logistic_regression', 'naive_bayes'] else 'Medium to High'}\n")
            f.write(f"- Expected Performance: {grade.title()} discriminatory power\n")
            f.write(f"- Risk Assessment: {'Low risk deployment' if grade in ['EXCELLENT', 'GOOD'] else 'Medium to high risk deployment'}\n")
        
        logger.info(f"Performance summary report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error creating performance summary: {e}")

def main():
    """Main model evaluation pipeline"""
    
    # Set up logging
    log_file = "model_evaluation.log"
    logger.add(log_file, rotation="500 MB", level="INFO")
    
    logger.info("="*80)
    logger.info("STARTING CREDIT SCORING MODEL EVALUATION PIPELINE")
    logger.info("="*80)
    
    # Check if custom modules are available
    if not IMPORTS_AVAILABLE:
        logger.error("Required modules not found. Please ensure model_fitting_engine.py and model_config.py exist.")
        return
    
    try:
        # Set project root
        project_root = set_project_root()
        
        # Define paths
        current_dir = os.getcwd()
        models_dir = "04.model_fitting\\outputs"
        test_data_dir = "03.binning_process\\outputs"
        output_dir = "05.model_evaluation\\outputs"
        
        test_file = "03.test_data.csv"
        test_path = os.path.join(current_dir, test_data_dir, test_file)
        models_path = os.path.join(current_dir, models_dir)
        
        # Create output directory
        create_directory_if_not_exists(os.path.join(current_dir, output_dir))
        output_folder = os.path.join(current_dir, output_dir)
        
        # Load test data
        logger.info("="*50)
        logger.info("STEP 1: LOADING TEST DATA")
        logger.info("="*50)
        
        test_df = load_data(test_path)
        
        # Prepare features and target
        selected_features = [

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
        # Verify features exist and prepare data
        available_features = [f for f in selected_features if f in test_df.columns]
        missing_features = [f for f in selected_features if f not in test_df.columns]
        
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
        
        X_test = test_df[available_features].values
        y_test = test_df['TARGET'].values
        
        logger.info(f"Test data prepared: {X_test.shape}")
        
        # Load trained models
        logger.info("="*50)
        logger.info("STEP 2: LOADING TRAINED MODELS")
        logger.info("="*50)
        
        models, scaler, best_model = load_trained_models(models_path)
        
        if not models:
            logger.error("No trained models found!")
            return
        
        logger.info(f"Loaded {len(models)} models")
        
        # Generate predictions
        logger.info("="*50)
        logger.info("STEP 3: GENERATING PREDICTIONS")
        logger.info("="*50)
        
        predictions_df = generate_predictions(models, X_test, y_test, scaler, output_folder)
        
        # Comprehensive evaluation
        logger.info("="*50)
        logger.info("STEP 4: COMPREHENSIVE MODEL EVALUATION")
        logger.info("="*50)
        
        evaluator = CreditScoringModelEvaluator()
        evaluation_results = evaluator.comprehensive_model_evaluation(
            models_dict=models,
            X_test=X_test,
            y_test=y_test,
            output_folder=output_folder,
            scaler=scaler
        )
        
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        print(evaluation_results.to_string(index=False))
        
        # Business scorecard analysis
        logger.info("="*50)
        logger.info("STEP 5: BUSINESS SCORECARD ANALYSIS")
        logger.info("="*50)
        
        if best_model:
            scorecard_stats = create_business_scorecard_analysis(
                best_model, X_test, y_test, scaler, output_folder
            )
            
            print("\n" + "="*60)
            print("BUSINESS SCORECARD ANALYSIS")
            print("="*60)
            print(scorecard_stats.to_string())

        # Scorecard evaluation
        logger.info("="*50)
        logger.info("STEP 5.5: SCORECARD PERFORMANCE EVALUATION")
        logger.info("="*50)
        
        scorecard_results = evaluate_scorecard_performance(output_folder, available_features)
        
        if scorecard_results:
            print("\n" + "="*60)
            print("SCORECARD PERFORMANCE RESULTS")
            print("="*60)
            print(f"AUC Score: {scorecard_results['auc_score']:.4f}")
            print(f"Gini Coefficient: {scorecard_results['gini']:.4f}")
            print("\nScore Band Statistics:")
            print(scorecard_results['band_stats'].to_string())
        else:
            print("No scorecard evaluation performed.")
        
        # Create comprehensive reports
        logger.info("="*50)
        logger.info("STEP 6: GENERATING REPORTS")
        logger.info("="*50)
        
        create_model_performance_summary(evaluation_results, output_folder)
        
        # Final summary
        logger.info("="*50)
        logger.info("STEP 7: FINAL SUMMARY")
        logger.info("="*50)
        
        best_result = evaluation_results.iloc[0]
        
        print("\n" + "="*80)
        print("MODEL EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best Model: {best_result['Model']}")
        print(f"ROC AUC: {best_result['ROC_AUC']:.4f}")
        print(f"Gini Coefficient: {best_result['Gini']:.4f}")
        print(f"KS Statistic: {best_result['KS_Statistic']:.4f}")
        print(f"Precision: {best_result['Precision']:.4f}")
        print(f"Recall: {best_result['Recall']:.4f}")
        print(f"F1 Score: {best_result['F1_Score']:.4f}")
        
        # Performance assessment
        auc_score = best_result['ROC_AUC']
        if auc_score >= PERFORMANCE_THRESHOLDS['excellent']['ROC_AUC']:
            assessment = "EXCELLENT - Deploy immediately"
        elif auc_score >= PERFORMANCE_THRESHOLDS['good']['ROC_AUC']:
            assessment = "GOOD - Ready for deployment"
        elif auc_score >= PERFORMANCE_THRESHOLDS['acceptable']['ROC_AUC']:
            assessment = "ACCEPTABLE - Consider deployment with monitoring"
        else:
            assessment = "NEEDS IMPROVEMENT - Do not deploy"
        
        print(f"\nPerformance Assessment: {assessment}")
        print(f"Models evaluated: {len(models)}")
        print(f"Test samples: {len(y_test)}")
        print(f"Output directory: {output_folder}")
        
        # Key output files
        print(f"\nKey Output Files:")
        print(f"- Evaluation results: {OUTPUT_CONFIG['model_files']['evaluation_results']}")
        print(f"- Detailed predictions: detailed_predictions.csv")
        print(f"- Business analysis: scorecard_analysis.csv")
        print(f"- Evaluation report: {OUTPUT_CONFIG['report_files']['evaluation_report']}")
        print(f"- Performance plots: Multiple visualization files")
        
        logger.info("Model evaluation pipeline completed successfully!")
        
        return {
            'evaluation_results': evaluation_results,
            'best_model_name': best_result['Model'],
            'predictions': predictions_df,
            'output_folder': output_folder
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation pipeline: {e}")
        raise

if __name__ == "__main__":
    main()