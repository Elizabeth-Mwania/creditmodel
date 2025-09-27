"""
Configuration file for credit scoring binning process
This file contains all configurable parameters for the binning process
"""

# Selection criteria for variables
SELECTION_CRITERIA = {
    "iv": {
        "min": 0.02,  # Minimum Information Value for variable selection
        "max": 0.8    # Maximum Information Value to avoid overfitting
    },
    "quality_score": {
        "min": 0.01   # Minimum quality score for variable selection
    }
}

# Global binning parameters
GLOBAL_BINNING_PARAMS = {
    "min_bin_size": 0.05,  # Minimum 5% of observations per bin
    "max_bins_default": 8  # Default maximum number of bins
}

# Data splitting parameters
DATA_SPLIT_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify_target": True
}

# Correlation analysis parameters
CORRELATION_PARAMS = {
    "threshold": 0.95,  # Correlation threshold for feature removal
    "method": "pearson"
}

# Variable-specific binning parameters
VARIABLE_BINNING_PARAMS = {
    # Transaction recency features
    "days_since_last_txn": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Higher days = higher risk
        "max_bins": 6,
        "description": "Days since last transaction"
    },
    
    "days_since_first_txn": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Longer history = lower risk
        "max_bins": 6,
        "description": "Days since first transaction (account age)"
    },
    
    "txn_days_range": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Wider range might indicate higher risk
        "max_bins": 6,
        "description": "Range of days between first and last transaction"
    },
    
    # Transaction volume features
    "txn_count_overall": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # More transactions = lower risk
        "max_bins": 8,
        "description": "Total transaction count"
    },
    
    "txn_count_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 8,
        "description": "Transaction count in last 6 months"
    },
    
    "txn_count_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Transaction count in last 3 months"
    },
    
    "txn_count_1m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Transaction count in last 1 month"
    },
    
    # Transaction amount features
    "sum_amount_overall": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Higher amounts = lower risk
        "max_bins": 6,
        "description": "Total transaction amount"
    },
    
    "sum_amount_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Sum of amounts in last 6 months"
    },
    
    "sum_amount_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Sum of amounts in last 3 months"
    },
    
    "sum_amount_1m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Sum of amounts in last 1 month"
    },
    
    # Average amount features
    "avg_amount_overall": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Average transaction amount (overall)"
    },
    
    "avg_amount_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Average amount in last 6 months"
    },
    
    "avg_amount_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Average amount in last 3 months"
    },
    
    "avg_amount_1m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Average amount in last 1 month"
    },
    
    # Amount range and variability features
    "max_amount_overall": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Maximum transaction amount"
    },
    
    "max_amount_6m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Maximum amount in last 6 months"
    },
    
    "min_amount_overall": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Minimum transaction amount"
    },
    
    "min_amount_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Minimum amount in last 6 months"
    },
    
    # Variability and consistency features
    "std_amount_overall": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Higher variability = higher risk
        "max_bins": 6,
        "description": "Standard deviation of amounts"
    },
    
    "std_amount_6m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Standard deviation of amounts (6m)"
    },
    
    "std_amount_3m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Standard deviation of amounts (3m)"
    },
    
    "std_amount_1m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Standard deviation of amounts (1m)"
    },
    
    "cv_amount_overall": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Higher coefficient of variation = higher risk
        "max_bins": 6,
        "description": "Coefficient of variation for amounts"
    },
    
    "cv_amount_6m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Coefficient of variation (6m)"
    },
    
    "cv_amount_3m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Coefficient of variation (3m)"
    },
    
    "cv_amount_1m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Coefficient of variation (1m)"
    },
    
    # Z-score based features (outlier detection)
    "mean_abs_z_overall": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Higher z-scores = more irregular behavior
        "max_bins": 6,
        "description": "Mean absolute z-score"
    },
    
    "mean_abs_z_6m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Mean absolute z-score (6m)"
    },
    
    "mean_abs_z_3m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Mean absolute z-score (3m)"
    },
    
    "mean_abs_z_1m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Mean absolute z-score (1m)"
    },
    
    "max_abs_z_overall": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Maximum absolute z-score"
    },
    
    "max_abs_z_6m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Maximum absolute z-score (6m)"
    },
    
    "max_abs_z_3m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Maximum absolute z-score (3m)"
    },
    
    # Activity pattern features
    "active_days_all": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # More active days = lower risk
        "max_bins": 6,
        "description": "Total number of active transaction days"
    },
    
    "active_days_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Active days in last 6 months"
    },
    
    "active_days_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Active days in last 3 months"
    },
    
    "active_days_1m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Active days in last 1 month"
    },
    
    # Transaction frequency features
    "avg_txn_per_day_overall": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Higher frequency = lower risk
        "max_bins": 8,
        "description": "Average transactions per day (overall)"
    },
    
    "avg_txn_per_day_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Average transactions per day (6m)"
    },
    
    "avg_txn_per_day_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Average transactions per day (3m)"
    },
    
    "avg_txn_per_day_1m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 8,
        "description": "Average transactions per day (1m)"
    },
    
    # Amount ratio features
    "amount_min_max_ratio_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Higher ratio = more consistent amounts
        "max_bins": 6,
        "description": "Min/Max amount ratio (6m)"
    },
    
    "amount_min_max_ratio_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Min/Max amount ratio (3m)"
    },
    
    "amount_min_max_ratio_1m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Min/Max amount ratio (1m)"
    },
    
    # Time-based features
    "avg_days_between_txn": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Longer gaps = higher risk
        "max_bins": 6,
        "description": "Average days between transactions"
    },
    
    # Growth and trend features
    "txn_count_growth_1m_vs_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Positive growth = lower risk
        "max_bins": 6,
        "description": "Transaction count growth (1m vs 3m)"
    },
    
    "txn_count_growth_3m_vs_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Transaction count growth (3m vs 6m)"
    },
    
    "sum_amount_growth_1m_vs_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Amount growth (1m vs 3m)"
    },
    
    "sum_amount_growth_3m_vs_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Amount growth (3m vs 6m)"
    },
    
    # Slope features
    "sum_amount_slope_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Positive slope = lower risk
        "max_bins": 6,
        "description": "Amount trend slope (6m)"
    },
    
    "txn_count_slope_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Transaction count slope (6m)"
    },
    
    "txn_slope_7d_1m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Need to verify direction
        "max_bins": 6,
        "description": "Transaction slope (7d to 1m)"
    },
    
    "txn_slope_1m_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Transaction slope (1m to 3m)"
    },
    
    # Dependency and ratio features
    "dependency_ratio_6m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Higher dependency = higher risk
        "max_bins": 6,
        "description": "Dependency ratio (6m)"
    },
    
    "dependency_ratio_3m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Dependency ratio (3m)"
    },
    
    "dependency_ratio_1m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Dependency ratio (1m)"
    },
    
    "dependency_ratio_overall": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Overall dependency ratio"
    },
    
    # Recent transaction features
    "rec_ratio_txn_7d_1m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Recent activity = lower risk
        "max_bins": 6,
        "description": "Recent transaction ratio (7d/1m)"
    },
    
    "rec_ratio_txn_1m_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Recent transaction ratio (1m/3m)"
    },
    
    # Transaction density features
    "txn_density_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Higher density = lower risk
        "max_bins": 6,
        "description": "Transaction density (3m)"
    },
    
    "txn_density_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Transaction density (6m)"
    },
    
    # Volatility and recency features
    "vol_recency_3m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Higher volatility recency = higher risk
        "max_bins": 6,
        "description": "Volatility recency (3m)"
    },
    
    "vol_recency_6m": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",
        "max_bins": 6,
        "description": "Volatility recency (6m)"
    },
    
    # Amount ratio features
    "avg_amt_ratio_1m_6m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Higher recent amounts = lower risk
        "max_bins": 6,
        "description": "Average amount ratio (1m/6m)"
    },
    
    "avg_amt_ratio_7d_3m": {
        "dtype": "numerical",
        "monotonic_trend": "descending",
        "max_bins": 6,
        "description": "Average amount ratio (7d/3m)"
    },
    
    # Loan-specific features
    "loan_type": {
        "dtype": "categorical",
        "max_bins": 5,
        "description": "Type of loan"
    },
    
    "loan_total_due": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Higher amounts = higher risk
        "max_bins": 6,
        "description": "Total loan amount due"
    },
    
    "loan_repaid_amounts": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # More repaid = lower risk
        "max_bins": 6,
        "description": "Amount repaid on loan"
    },
    
    "loan_repayment_rate": {
        "dtype": "numerical",
        "monotonic_trend": "descending",  # Higher repayment rate = lower risk
        "max_bins": 6,
        "description": "Loan repayment rate (%)"
    },
    
    "loan_repay_days": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # More days = higher risk
        "max_bins": 8,
        "description": "Days to repay loan"
    },
    
    "loan_shortfall": {
        "dtype": "numerical",
        "monotonic_trend": "ascending",  # Higher shortfall = higher risk
        "max_bins": 6,
        "description": "Loan shortfall amount"
    }
}

# Output file paths and naming conventions
OUTPUT_CONFIG = {
    "train_filename": "03.train_data.csv",
    "test_filename": "03.test_data.csv",
    "iv_table_filename": "iv_table.csv",
    "binning_report_filename": "binning_process_report.txt",
    "log_filename": "binning_process.log",
    "plot_format": "png",
    "plot_dpi": 300
}

# Feature engineering validation rules
VALIDATION_RULES = {
    "min_unique_values": 2,  # Minimum unique values for a feature to be considered
    "max_missing_rate": 0.8,  # Maximum missing value rate (80%)
    "min_observations_per_bin": 50,  # Minimum observations per bin
    "max_correlation": 0.95  # Maximum correlation for feature removal
}

# Business rules and constraints
BUSINESS_RULES = {
    "min_iv_for_strong_predictor": 0.30,  # Strong predictive power
    "min_iv_for_moderate_predictor": 0.10,  # Moderate predictive power
    "max_bins_for_categorical": 10,  # Maximum bins for categorical variables
    "monotonicity_tolerance": 0.05,  # Tolerance for monotonicity violations
    "target_classes": [0, 1],  # Expected target variable classes
    "positive_class": 1  # Class representing "bad" customers
}

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    "min_auc": 0.60,  # Minimum acceptable AUC
    "min_ks": 0.20,  # Minimum acceptable KS statistic
    "max_feature_count": 20,  # Maximum number of features in final model
    "min_feature_count": 5   # Minimum number of features in final model
}

def get_variable_params(variable_name: str) -> dict:
    """
    Get binning parameters for a specific variable
    
    Parameters:
    - variable_name: Name of the variable
    
    Returns:
    - Dictionary of parameters for the variable
    """
    return VARIABLE_BINNING_PARAMS.get(variable_name, {
        "dtype": "numerical",
        "max_bins": GLOBAL_BINNING_PARAMS["max_bins_default"],
        "description": f"Auto-detected parameter for {variable_name}"
    })

def get_all_configured_variables() -> list:
    """Get list of all variables with configured parameters"""
    return list(VARIABLE_BINNING_PARAMS.keys())

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Validate IV criteria
    if SELECTION_CRITERIA["iv"]["min"] >= SELECTION_CRITERIA["iv"]["max"]:
        errors.append("IV minimum must be less than IV maximum")
    
    # Validate bin size
    if not 0 < GLOBAL_BINNING_PARAMS["min_bin_size"] < 1:
        errors.append("min_bin_size must be between 0 and 1")
    
    # Validate test size
    if not 0 < DATA_SPLIT_PARAMS["test_size"] < 1:
        errors.append("test_size must be between 0 and 1")
    
    # Check for valid monotonic trends
    valid_trends = [None, "ascending", "descending"]
    for var, params in VARIABLE_BINNING_PARAMS.items():
        if params.get("monotonic_trend") not in valid_trends:
            errors.append(f"Invalid monotonic_trend for {var}")
    
    if errors:
        raise ValueError("Configuration validation failed: " + "; ".join(errors))
    
    return True

def print_config_summary():
    """Print a summary of the configuration"""
    print("=== BINNING CONFIGURATION SUMMARY ===")
    print(f"Total configured variables: {len(VARIABLE_BINNING_PARAMS)}")
    print(f"Selection criteria - Min IV: {SELECTION_CRITERIA['iv']['min']}")
    print(f"Selection criteria - Max IV: {SELECTION_CRITERIA['iv']['max']}")
    print(f"Minimum bin size: {GLOBAL_BINNING_PARAMS['min_bin_size']*100}%")
    print(f"Test set size: {DATA_SPLIT_PARAMS['test_size']*100}%")
    
    # Count by data type
    numerical_count = sum(1 for p in VARIABLE_BINNING_PARAMS.values() if p.get('dtype') == 'numerical')
    categorical_count = sum(1 for p in VARIABLE_BINNING_PARAMS.values() if p.get('dtype') == 'categorical')
    
    print(f"Numerical variables: {numerical_count}")
    print(f"Categorical variables: {categorical_count}")
    
    # Count by monotonic trend
    ascending_count = sum(1 for p in VARIABLE_BINNING_PARAMS.values() if p.get('monotonic_trend') == 'ascending')
    descending_count = sum(1 for p in VARIABLE_BINNING_PARAMS.values() if p.get('monotonic_trend') == 'descending')
    no_trend_count = sum(1 for p in VARIABLE_BINNING_PARAMS.values() if p.get('monotonic_trend') is None)
    
    print(f"Ascending trend variables: {ascending_count}")
    print(f"Descending trend variables: {descending_count}")
    print(f"No trend specified: {no_trend_count}")

# Validate configuration on import
if __name__ == "__main__":
    validate_config()
    print_config_summary()
    print("Configuration validation passed!")