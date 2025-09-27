import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_gbq
from datetime import datetime, timedelta
import warnings
import os
import sys
from loguru import logger



# Suppress warnings
warnings.filterwarnings("ignore")


def set_project_root():
    """
    Set the working directory to the project root by moving up the directory tree
    until a known project-level file/folder (e.g., '.git') is found.

    Returns:
        str: The absolute path to the project root.
    """
    current_file_path = os.path.abspath(__file__)
    project_root = current_file_path

    while not os.path.exists(
            os.path.join(project_root, ".git")):  
        project_root = os.path.dirname(project_root)

    os.chdir(project_root)
    print("Current working directory:", os.getcwd())
    return

def create_directory_if_not_exists(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory created: {path}")



def load_model_data():
    """
    Load model data from Feature store and create good_bad target column.

    Returns:
        pd.DataFrame: Model data with target column.
    """
    try:
         # fetch data from data_for_preprocessing.csv
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(base_path, "00.data", "outputs", "data_for_preprocessing.csv")
        
        logger.info(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        # Load data
        model_data = pd.read_csv(data_path)
        logger.info(f"Loaded data shape: {model_data.shape}")
        
        model_data['customer_id'] = model_data['customer_id'].astype('string')

        # Define good (0) vs bad (1)
        model_data['TARGET'] = (
            (model_data['loan_repayment_rate'] >= 100) &
            (model_data['loan_repay_days'] <= 10)
        ).astype(int)

        # Invert so that good=0, bad=1
        model_data['TARGET'] = model_data['TARGET'].replace({1: 0, 0: 1})

        print("Model data shape:", model_data.shape)
        print("Target distribution:\n", model_data['TARGET'].value_counts(normalize=True))

        return model_data
    except Exception as e:
        print(f"Error loading model data: {e}")
        return pd.DataFrame()

def save_data_for_eda(df):
    """
    Save the data to a CSV file for Exploratory Data Analysis (EDA).
    """
    try:
        current_dir = os.getcwd()
        subdirectory = "01.data_processing\\outputs"
        filename = "01.data.csv"

        # Create the directory if it doesn't exist
        create_directory_if_not_exists(os.path.join(current_dir, subdirectory))

        file_path = os.path.join(current_dir, subdirectory, filename)
        df.to_csv(path_or_buf=file_path, index=False)
        print(f"Data successfully saved for EDA: {file_path}")
    except Exception as e:
        print(f"Error writing model data for EDA: {e}")


def main():
    # Set project root
    set_project_root()

    # Load model data with target
    model_data = load_model_data()
    if model_data.empty:
        print("No model data available. Exiting...")
        return

    # Save data for EDA
    save_data_for_eda(model_data)


if __name__ == "__main__":
    main()  

     