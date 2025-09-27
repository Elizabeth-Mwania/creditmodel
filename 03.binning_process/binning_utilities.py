"""
Utility functions for credit scoring binning process
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from loguru import logger
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_iv_strength(iv_value: float) -> str:
    """
    Classify Information Value strength based on standard ranges
    
    Parameters:
    - iv_value: Information Value
    
    Returns:
    - String classification of IV strength
    """
    if iv_value < 0.02:
        return "Not useful"
    elif iv_value < 0.10:
        return "Weak"
    elif iv_value < 0.30:
        return "Medium"
    elif iv_value < 0.50:
        return "Strong"
    else:
        return "Suspicious"

def calculate_psi(expected: np.array, actual: np.array, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI)
    
    Parameters:
    - expected: Expected distribution (typically training set)
    - actual: Actual distribution (typically validation/test set)
    - bins: Number of bins for calculation
    
    Returns:
    - PSI value
    """
    try:
        # Create bins based on expected distribution
        breakpoints = np.linspace(0, 1, bins + 1)
        breakpoints = np.quantile(expected, breakpoints)
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) < 2:
            return float('inf')
        
        # Calculate expected percentages
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        expected_percents = np.clip(expected_percents, 1e-4, 1)  # Avoid division by zero
        
        # Calculate actual percentages
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        actual_percents = np.clip(actual_percents, 1e-4, 1)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return psi
    
    except Exception as e:
        logger.warning(f"Error calculating PSI: {e}")
        return float('inf')

def interpret_psi(psi_value: float) -> str:
    """Interpret PSI value"""
    if psi_value < 0.10:
        return "No significant change"
    elif psi_value < 0.25:
        return "Minor change"
    else:
        return "Major change - investigate"

def calculate_gini_coefficient(y_true: np.array, y_prob: np.array) -> float:
    """Calculate Gini coefficient from probabilities"""
    try:
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        return gini
    except Exception as e:
        logger.warning(f"Error calculating Gini: {e}")
        return 0.0

def calculate_ks_statistic(y_true: np.array, y_prob: np.array) -> float:
    """Calculate Kolmogorov-Smirnov statistic"""
    try:
        # Separate good and bad customers
        good_scores = y_prob[y_true == 0]
        bad_scores = y_prob[y_true == 1]
        
        # Calculate cumulative distributions
        score_range = np.linspace(0, 1, 100)
        good_cdf = [np.mean(good_scores <= score) for score in score_range]
        bad_cdf = [np.mean(bad_scores <= score) for score in score_range]
        
        # Calculate KS statistic
        ks = max(abs(np.array(good_cdf) - np.array(bad_cdf)))
        return ks
    
    except Exception as e:
        logger.warning(f"Error calculating KS: {e}")
        return 0.0

def create_scorecard_points(woe_dict: Dict, base_score: int = 600, pdo: int = 20) -> Dict:
    """
    Convert WoE values to scorecard points
    
    Parameters:
    - woe_dict: Dictionary mapping bins to WoE values
    - base_score: Base score (default 600)
    - pdo: Points to double the odds (default 20)
    
    Returns:
    - Dictionary mapping bins to points
    """
    # Calculate factor and offset
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(20)  # Assuming 20:1 odds at base score
    
    points_dict = {}
    for bin_value, woe in woe_dict.items():
        points = offset + factor * woe
        points_dict[bin_value] = round(points)
    
    return points_dict

def validate_monotonicity(woe_values: List[float], trend: str) -> Tuple[bool, float]:
    """
    Validate if WoE values follow monotonic trend
    
    Parameters:
    - woe_values: List of WoE values
    - trend: 'ascending' or 'descending'
    
    Returns:
    - Tuple of (is_monotonic, violation_rate)
    """
    if len(woe_values) < 2:
        return True, 0.0
    
    violations = 0
    total_comparisons = len(woe_values) - 1
    
    for i in range(1, len(woe_values)):
        if trend == 'ascending':
            if woe_values[i] < woe_values[i-1]:
                violations += 1
        elif trend == 'descending':
            if woe_values[i] > woe_values[i-1]:
                violations += 1
    
    violation_rate = violations / total_comparisons if total_comparisons > 0 else 0
    is_monotonic = violation_rate == 0
    
    return is_monotonic, violation_rate

def create_feature_importance_plot(feature_scores: Dict[str, float], 
                                 title: str = "Feature Importance (Information Value)",
                                 save_path: str = None,
                                 top_n: int = 20) -> None:
    """
    Create feature importance plot based on IV scores
    
    Parameters:
    - feature_scores: Dictionary of feature names and their IV scores
    - title: Plot title
    - save_path: Path to save plot
    - top_n: Number of top features to show
    """
    # Sort features by IV score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_features:
        logger.warning("No features to plot")
        return
    
    features, scores = zip(*sorted_features)
    
    # Create plot
    plt.figure(figsize=(12, max(6, len(features) * 0.4)))
    colors = ['red' if score < 0.02 else 'orange' if score < 0.10 else 'yellow' if score < 0.30 else 'green' for score in scores]
    
    bars = plt.barh(range(len(features)), scores, color=colors, alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Information Value')
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    
    # Add IV strength annotations
    for i, (feature, score) in enumerate(sorted_features):
        strength = calculate_iv_strength(score)
        plt.text(score + max(scores) * 0.01, i, f'{score:.3f} ({strength})', 
                va='center', ha='left', fontsize=8)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Not useful (<0.02)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.7, label='Weak (0.02-0.10)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.7, label='Medium (0.10-0.30)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='Strong (>0.30)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()

def create_correlation_heatmap(df: pd.DataFrame, 
                              title: str = "Feature Correlation Matrix",
                              save_path: str = None,
                              threshold: float = 0.7) -> None:
    """
    Create correlation heatmap for numerical features
    
    Parameters:
    - df: DataFrame with numerical features
    - title: Plot title
    - save_path: Path to save plot
    - threshold: Correlation threshold to highlight
    """
    # Select numerical columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'customer_id' in numeric_cols:
        numeric_cols.remove('customer_id')
    if 'TARGET' in numeric_cols:
        numeric_cols.remove('TARGET')
    
    if len(numeric_cols) < 2:
        logger.warning("Not enough numerical features for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask for high correlations
    mask = np.abs(corr_matrix) < threshold
    
    # Create plot
    plt.figure(figsize=(max(8, len(numeric_cols) * 0.5), max(6, len(numeric_cols) * 0.4)))
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                mask=None,
                annot=False,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Correlation heatmap saved to {save_path}")
    else:
        plt.show()
    
    # Log high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs (>={threshold}):")
        for var1, var2, corr_val in high_corr_pairs:
            logger.info(f"  {var1} - {var2}: {corr_val:.3f}")

def create_target_distribution_plot(y: np.array, 
                                  title: str = "Target Variable Distribution",
                                  save_path: str = None) -> None:
    """Create target variable distribution plot"""
    plt.figure(figsize=(8, 5))
    
    # Count plot
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts, alpha=0.7, color=['green', 'red'])
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.title('Target Count Distribution')
    for i, (val, count) in enumerate(zip(unique, counts)):
        plt.text(val, count + max(counts) * 0.01, str(count), ha='center')
    
    # Percentage plot
    plt.subplot(1, 2, 2)
    percentages = counts / len(y) * 100
    plt.bar(unique, percentages, alpha=0.7, color=['green', 'red'])
    plt.xlabel('Target')
    plt.ylabel('Percentage (%)')
    plt.title('Target Percentage Distribution')
    for i, (val, pct) in enumerate(zip(unique, percentages)):
        plt.text(val, pct + max(percentages) * 0.01, f'{pct:.1f}%', ha='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Target distribution plot saved to {save_path}")
    else:
        plt.show()

def export_binning_summary_to_excel(binning_process, output_path: str) -> None:
    """
    Export comprehensive binning summary to Excel file
    
    Parameters:
    - binning_process: Fitted CreditScoringBinningProcess object
    - output_path: Path for Excel file
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = binning_process.summary()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual variable sheets
            for var_name, binning in binning_process.fitted_binnings.items():
                if binning.is_fitted:
                    var_summary = binning.get_binning_summary()
                    sheet_name = var_name[:31]  # Excel sheet name limit
                    var_summary.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Binning summary exported to Excel: {output_path}")
    
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")

def create_model_performance_report(y_true: np.array, y_pred: np.array, y_prob: np.array = None) -> Dict:
    """
    Create comprehensive model performance report
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_prob: Predicted probabilities (optional)
    
    Returns:
    - Dictionary with performance metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, confusion_matrix, classification_report)
    
    performance = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        performance['auc'] = roc_auc_score(y_true, y_prob)
        performance['gini'] = calculate_gini_coefficient(y_true, y_prob)
        performance['ks_statistic'] = calculate_ks_statistic(y_true, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    performance['confusion_matrix'] = cm.tolist()
    
    # Classification report
    performance['classification_report'] = classification_report(y_true, y_pred)
    
    return performance

def detect_outliers_iqr(data: np.array, multiplier: float = 1.5) -> Tuple[np.array, np.array]:
    """
    Detect outliers using Interquartile Range method
    
    Parameters:
    - data: Input data
    - multiplier: IQR multiplier for outlier detection
    
    Returns:
    - Tuple of (outlier_indices, outlier_values)
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    outlier_values = data[outlier_mask]
    
    return outlier_indices, outlier_values

def winsorize_outliers(data: np.array, limits: Tuple[float, float] = (0.01, 0.01)) -> np.array:
    """
    Winsorize outliers by clipping to specified percentiles
    
    Parameters:
    - data: Input data
    - limits: Lower and upper percentile limits for clipping
    
    Returns:
    - Winsorized data
    """
    return stats.mstats.winsorize(data, limits=limits)

def create_diagnostic_plots(binning_process, output_folder: str) -> None:
    """
    Create diagnostic plots for binning process
    
    Parameters:
    - binning_process: Fitted binning process
    - output_folder: Folder to save plots
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Feature importance plot
    iv_scores = {var: binning.iv for var, binning in binning_process.fitted_binnings.items() if binning.is_fitted}
    
    if iv_scores:
        create_feature_importance_plot(
            iv_scores, 
            save_path=os.path.join(output_folder, "feature_importance.png")
        )
    
    logger.info(f"Diagnostic plots saved to {output_folder}")
