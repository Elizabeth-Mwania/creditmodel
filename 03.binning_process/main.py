import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from loguru import logger
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
from datetime import datetime

# Import necessary modules
try:
    from credit_binning_process import (
        CreditScoringBinning, 
        CreditScoringBinningProcess, 
        woe_iv_filter, 
        generate_woe_plots,
        create_directory_if_not_exists
    )
    from binning_config import (
        VARIABLE_BINNING_PARAMS,
        SELECTION_CRITERIA,
        GLOBAL_BINNING_PARAMS,
        DATA_SPLIT_PARAMS,
        OUTPUT_CONFIG,
        get_variable_params,
        validate_config,
        print_config_summary
    )
    from binning_utilities import (
        calculate_iv_strength,
        create_feature_importance_plot,
        create_correlation_heatmap,
        create_target_distribution_plot,
        export_binning_summary_to_excel,
        create_diagnostic_plots,
        run_analysis
    )
    IMPORTS_AVAILABLE = True
    print("All modules imported successfully!")
    
except ImportError as e:
    print(f"Modules not found: {e}")
    print("The following files are required:")
    print("- credit_binning_process.py")
    print("- binning_config.py") 
    print("- binning_utilities.py")
    IMPORTS_AVAILABLE = False

pd.options.display.max_columns = None
pd.options.display.max_rows = None
warnings.filterwarnings('ignore')
color = sns.color_palette()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format

plt.style.use('ggplot')
sns.set_style({'axes.grid': False})


def set_project_root():
    """
    Set the working directory to the project root by moving up the directory tree
    until a known project-level file/folder (e.g., '.git') is found.

    Returns:
        str: The absolute path to the project root.
    """
    current_file_path = os.path.abspath(__file__)
    project_root = current_file_path

    while not os.path.exists(os.path.join(project_root, ".git")):
        project_root = os.path.dirname(project_root)
        # Prevent infinite loop
        if project_root == os.path.dirname(project_root):
            project_root = os.getcwd()
            break

    os.chdir(project_root)
    print("Current working directory:", os.getcwd())
    return project_root


def create_directory_if_not_exists_fallback(path):
    """Fallback function if modules aren't available"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")


def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath_or_buffer=file_path)
        print("Data shape: ", df.shape)
        print("Good / bad ratio: ", df['TARGET'].value_counts(normalize=True))
        print("Full list of columns:")
        for col in df.columns:
            print(col)
        logger.info(f"Data loaded successfully from {file_path}.")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def custom_train_test_split(df, test_size=0.3, random_state=42, stratify=None):
    """
    Splits the data into train and test sets, ready for modeling.
    """
    # Split into train and test sets
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    
    # Display the distribution of the target variable in both sets
    print("Train TARGET distribution:")
    print(train['TARGET'].value_counts(normalize=True))
    print("\nTest TARGET distribution:")
    print(test['TARGET'].value_counts(normalize=True))
    
    return train, test


def data_quality_check(train, test):
    """
    Perform data quality checks
    """
    print("=== DATA QUALITY CHECK ===")
    
    # Check for infinite values
    print("Checking for infinite values...")
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check missing values
    train_missing = train.isnull().sum()
    test_missing = test.isnull().sum()
    
    print("Missing values in train set:")
    print(train_missing[train_missing > 0])
    print("\nMissing values in test set:")
    print(test_missing[test_missing > 0])
    
    # Check for constant features
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['customer_id', 'merchant_id', 'TARGET']]
    
    constant_features = []
    for col in numeric_cols:
        if train[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"Removing constant features: {constant_features}")
        train = train.drop(columns=constant_features)
        test = test.drop(columns=constant_features)
    
    # Check for quasi-constant features (>95% same value)
    quasi_constant_features = []
    for col in numeric_cols:
        if col in train.columns:  # Check if column still exists after constant removal
            if train[col].value_counts().iloc[0] / len(train) > 0.95:
                quasi_constant_features.append(col)
    
    if quasi_constant_features:
        print(f"Found quasi-constant features (>95% same value): {quasi_constant_features}")
    
    print(f"Final train shape: {train.shape}")
    print(f"Final test shape: {test.shape}")
    
    return train, test


def save_train_test_data(train, test, output_folder):
    """Save the processed data to CSV files for the next step."""
    try:
        # Use fallback directory creation if modules not available
        if IMPORTS_AVAILABLE:
            create_directory_if_not_exists(output_folder)
        else:
            create_directory_if_not_exists_fallback(output_folder)
        
        train_path = os.path.join(output_folder, "03.train_data.csv")
        test_path = os.path.join(output_folder, "03.test_data.csv")
        
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        
        logger.info(f"Train data saved to: {train_path}")
        logger.info(f"Test data saved to: {test_path}")
        print("Train and test data successfully saved for modeling.")
        
    except Exception as e:
        logger.error(f"Error saving train/test data: {e}")
        raise


def fallback_binning_params():
    """Fallback binning parameters if config module not available"""
    return {
        "days_since_last_txn": {
            "dtype": "numerical",
            "monotonic_trend": "ascending",
            "max_bins": 8
        },
        "days_since_first_txn": {
            "dtype": "numerical", 
            "monotonic_trend": "descending",
            "max_bins": 8
        },
        "txn_count_overall": {
            "dtype": "numerical",
            "monotonic_trend": "descending", 
            "max_bins": 8
        },
        "sum_amount_3m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "avg_amount_1m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "loan_type": {
            "dtype": "categorical",
            "max_bins": 5
        }
    }


def create_simple_report(selected_vars, output_folder):
    """When the modules aren't available"""
    try:
        report_path = os.path.join(output_folder, "report","binning_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BINNING PROCESS REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Selected variables: {len(selected_vars)}\n\n")
            f.write("Variables:\n")
            for i, var in enumerate(selected_vars, 1):
                f.write(f"{i}. {var}\n")
        
        print(f"Simple report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error creating simple report: {e}")


def main():
    """Main execution function"""
    # Set up logging
    logger.add("binning_process.log", rotation="500 MB", level="INFO")
    logger.info("Starting credit scoring binning process")
    
    # Check if custom modules are available
    if not IMPORTS_AVAILABLE:
        print("\n" + "="*60)
        print("WARNING: Modules not found!")
        print("Running in fallback mode with basic functionality.")
        print("To get full functionality, please create the required files.")
        print("="*60)
    
    # Set the project root directory
    try:
        project_root = set_project_root()
    except Exception as e:
        print(f"Could not find .git folder, using current directory: {e}")
        project_root = os.getcwd()
    
    # Define paths
    current_dir = os.getcwd()
    input_subdirectory = "02.eda\\outputs"
    output_subdirectory = "03.binning_process\\outputs"
    
    input_filename = "02.data.csv"
    input_file_path = os.path.join(current_dir, input_subdirectory, input_filename)
    output_folder = os.path.join(current_dir, output_subdirectory)
    
    # Create output directory
    if IMPORTS_AVAILABLE:
        create_directory_if_not_exists(output_folder)
    else:
        create_directory_if_not_exists_fallback(output_folder)
    
    try:
        # Load data
        logger.info("Loading data...")
        data = load_data(input_file_path)
        
        # Split into train and test sets
        logger.info("Splitting data into train and test sets...")
        if IMPORTS_AVAILABLE:
            test_size = DATA_SPLIT_PARAMS["test_size"]
            random_state = DATA_SPLIT_PARAMS["random_state"] 
        else:
            test_size = 0.2
            random_state = 42
            
        train, test = custom_train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=data['TARGET']
        )
        
        # Data quality checks
        logger.info("Performing data quality checks...")
        train, test = data_quality_check(train, test)
        
        if IMPORTS_AVAILABLE:
            # Full functionality with custom modules
            logger.info("Running full binning process...")
            
            # Validate configuration
            try:
                validate_config()
                print_config_summary()
            except Exception as e:
                logger.warning(f"Config validation failed: {e}")
            
            # Run analysis
            try:
                run_analysis(data, target_col='TARGET')
            except Exception as e:
                logger.warning(f" Analysis failed: {e}")
            
            # Apply binning process
            available_vars = [col for col in train.columns if col not in ['customer_id', 'merchant_id', 'TARGET']]
            configured_vars = list(VARIABLE_BINNING_PARAMS.keys())
            
            final_binning_params = {}
            for var in available_vars:
                if var in configured_vars:
                    final_binning_params[var] = VARIABLE_BINNING_PARAMS[var]
                else:
                    final_binning_params[var] = {
                        "dtype": "numerical",
                        "max_bins": GLOBAL_BINNING_PARAMS["max_bins_default"]
                    }
            
            logger.info(f"Processing {len(final_binning_params)} variables for binning")
            
            # Apply WoE/IV filtering and binning
            train_filtered, test_filtered, binning_process = woe_iv_filter(
                train, test, final_binning_params, output_folder=output_folder
            )
            
            # Generate visualizations and reports
            generate_woe_plots(train_filtered, binning_process, output_folder)
            
            # Get selected variables
            selected_vars = binning_process.get_selected_variables()
            
            # Create comprehensive reports
            try:
                export_binning_summary_to_excel(binning_process, os.path.join(output_folder, "binning_summary.xlsx"))
            except Exception as e:
                logger.warning(f"Excel export failed: {e}")
                
        else:
            # Fallback mode, basic functionality only
            logger.info("Running in fallback mode...")
            
            # Use fallback parameters
            binning_params = fallback_binning_params()
            
            # Filter to only available variables
            available_vars = [col for col in train.columns if col not in ['customer_id', 'merchant_id', 'TARGET']]
            available_configured_vars = [var for var in binning_params.keys() if var in available_vars]
            
            # For fallback mode, just use available variables
            train_filtered = train[['customer_id'] + available_configured_vars + ['TARGET']].copy()
            test_filtered = test[['customer_id'] + available_configured_vars + ['TARGET']].copy()
            
            selected_vars = available_configured_vars
            
            print(f"Fallback mode: Selected {len(selected_vars)} variables")
        
        # Save processed data
        logger.info("Saving processed data...")
        save_train_test_data(train_filtered, test_filtered, output_folder)
        
        # Create report
        if IMPORTS_AVAILABLE:
            # Use custom report function 
            pass
        else:
            create_simple_report(selected_vars, output_folder)
        
        # Print final summary
        print("\n" + "="*60)
        print("BINNING PROCESS COMPLETED!")
        print("="*60)
        print(f"Original dataset shape: {data.shape}")
        print(f"Final train set shape: {train_filtered.shape}")
        print(f"Final test set shape: {test_filtered.shape}")
        print(f"Variables selected for modeling: {len(selected_vars)}")
        print(f"Output saved to: {output_folder}")
        
        # Display selected variables
        print(f"\nSelected variables ({len(selected_vars)}):")
        for i, var in enumerate(selected_vars, 1):
            print(f"{i:2d}. {var}")
        
        logger.info("Binning process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()