import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from datetime import date
import warnings
import os
from loguru import logger
import statsmodels.api as sm
from scipy.stats import ttest_ind
from feature_engineering import generate_features 
warnings.filterwarnings("ignore")
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


def missing_values_table(df):
    """
    Generates a table showing the number and percentage of missing values in each column.

    Parameters:
    - df: pd.DataFrame, the DataFrame to analyze.

    Returns:
    - pd.DataFrame, a DataFrame with missing values summary.
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(
        mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# load data from step 01
def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath_or_buffer=file_path)
        print("Data shape: ", df.shape)

        if "TARGET" in df.columns:
            print("Good / bad ratio: ", df['TARGET'].value_counts(normalize=True))
        else:
            logger.warning("'TARGET' column missing in dataset.")

        print("Full list of columns:")
        for col in df.columns:
            print(col)
        logger.info(f"Data loaded successfully from {file_path}.")
        return df

    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()   # return empty DataFrame instead of None



## handling missing values
def handle_missing_values(df, threshold=90, method='drop'):
    """
    Handles missing values in a DataFrame based on a threshold.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to process.
    - threshold: float, the percentage threshold of missing values to determine action.
    - method: str, the method to handle missing values ('drop' or 'fill').
    
    Returns:
    - pd.DataFrame, the DataFrame with missing values handled.
    """
    # Get the missing values table
    mis_val_table = missing_values_table(df)
    print(mis_val_table.to_string())

    # Determine columns to drop based on threshold
    cols_to_drop = mis_val_table[mis_val_table['% of Total Values'] > threshold].index.tolist()
    if method == 'drop':
        # Drop columns with missing values above the threshold
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    elif method == 'fill':
        # Fill missing values with the median (or any other method)
        for col in cols_to_drop:
            df[col] = df[col].fillna(df[col].median())
        print(f"Filled missing values in columns: {cols_to_drop}")

    # Handle the remaining columns with missing values
    remaining_cols = mis_val_table[mis_val_table['% of Total Values'] <= threshold].index.tolist()
    df[remaining_cols] = df[remaining_cols].replace([np.inf, -np.inf, 'NA', 'N/A', '', 'NaT'], np.nan)
    df = df.fillna(0)
    print(f"Filled missing values in remaining columns")
    return df


def feature_summary(df_fa, output_folder):
    """
    Generates a summary of the features in the provided DataFrame, including information 
    on null values, unique counts, data types, and statistical measures for numerical columns.

    Parameters:
    - df_fa: pd.DataFrame, the DataFrame containing the data to summarize.

    Returns:
    - pd.DataFrame, a summary DataFrame with details on null values, unique counts, data types,
      max/min values, mean, standard deviation, and skewness for each feature.
    """
    create_directory_if_not_exists(output_folder)
    print('DataFrame shape')
    print('rows:', df_fa.shape[0])
    print('cols:', df_fa.shape[1])
    col_list = ['Null', 'Unique_Count', 'Data_type', 'Max/Min', 'Mean', 'Std', 'Skewness',  # 'Sample_values'
                ]
    df = pd.DataFrame(index=df_fa.columns, columns=col_list)
    df['Null'] = list([len(df_fa[col][df_fa[col].isnull()]) for i, col in enumerate(df_fa.columns)])
    df['Unique_Count'] = list([len(df_fa[col].unique()) for i, col in enumerate(df_fa.columns)])
    df['Data_type'] = list([df_fa[col].dtype for i, col in enumerate(df_fa.columns)])
    for i, col in enumerate(df_fa.columns):
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col, 'Max/Min'] = str(round(df_fa[col].max(), 2)) + '/' + str(round(df_fa[col].min(), 2))
            df.at[col, 'Mean'] = df_fa[col].mean()
            df.at[col, 'Std'] = df_fa[col].std()
            df.at[col, 'Skewness'] = df_fa[col].skew()

    df = df.fillna('-')

    # Set display options for better readability
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Set width to avoid column breaking
    pd.set_option('display.float_format', '{:.2f}'.format)  # Format float values

    print(df.to_string())

    output_csv_file = os.path.join(output_folder, "summary_statistics.csv")
    df = df.reset_index()
    df = df.rename(columns={'index': 'Variable'})
    # Save the summary to a CSV file
    df.to_csv(output_csv_file, index=False)
    print(f"Binning summary saved to {output_csv_file}")

    return df

def plot_bivariate_distributions(df, columns, target_column, output_folder):
    """Plot each variable distribution to the target variable"""

    create_directory_if_not_exists(output_folder)

 
    numeric_cols = df[columns].select_dtypes(include=['float64', 'int64']).columns

    fig = plt.figure(figsize=(18, 24))
    for i, column in enumerate(numeric_cols):
        plt.subplot(-(-len(numeric_cols) // 2), 2, i + 1)
        sns.kdeplot(
            data=df,
            x=column,
            hue=target_column,
            fill=True,
            common_norm=False,
            alpha=0.5
        )
        plt.title(f"Distribution of {column} by {target_column}")

    plt.tight_layout()
    dist_path = os.path.join(output_folder, "bivariate_distribution.png")
    plt.savefig(dist_path)
    plt.close()
    return


def bivariate_analysis(data, output_folder):
    create_directory_if_not_exists(output_folder)

    numerical_features = data.select_dtypes(include=['float64', 'int64']).drop(columns=['TARGET']).columns

    for col in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x="TARGET", y=col, data=data)
        plt.title(f'{col} vs Target')
        dist_path = os.path.join(output_folder, f"{col} bivariate analysis.png")
        plt.savefig(dist_path)
        plt.close()
    return


def distribution_plot(data, output_folder):
    """
    Performs exploratory data analysis (EDA) on the provided DataFrame by plotting 
    the distribution and count of the target variable.

    Parameters:
    - data: pd.DataFrame, the input DataFrame containing the target variable.

    Returns:
    - None, but displays the target distribution plots.
    """
    # plot target variable distribution
    plt.figure(figsize=(13, 4))
    plt.subplot(121)
    data.TARGET.value_counts().plot.pie(autopct="%1.0f%%", colors=sns.color_palette("prism", 7), startangle=60,
                                        labels=["good", "bad"],
                                        wedgeprops={"linewidth": 2, "edgecolor": "k"}, explode=[.1, 0], shadow=True)
    plt.title("Distribution of target variable")

    plt.subplot(122)
    ax = data.TARGET.value_counts().plot(kind="barh")

    for i, j in enumerate(data.TARGET.value_counts().values):
        ax.text(.7, i, j, weight="bold", fontsize=20)

    plt.title("Count of target variable")
    dist_path = os.path.join(output_folder, "distribution.png")
    plt.savefig(dist_path)
    plt.close()


def handling_contant_and_quasi_contant(data):
    """
    Identifies and removes constant, quasi-constant, and duplicate features from 
    the provided DataFrame to improve model performance.

    Parameters:
    - data: pd.DataFrame, the input DataFrame to process.

    Returns:
    - pd.DataFrame, the DataFrame with constant, quasi-constant, and duplicate features removed.
    """
    # checking for consant variables

    mydata_const_cols = [c for c in data.columns if len(data[c].unique()) == 1]
    print("{} columns with a unique value on mydata set".format(len(mydata_const_cols)))
    print("constant features:", mydata_const_cols)
    numeric_data = data.select_dtypes(include=np.number)
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.fillna(0)

    # checking for quasi constant variables
    qconstant_filter = VarianceThreshold(threshold=0.01)
    qconstant_filter.fit(numeric_data)
    qconstant_columns = [column for column in numeric_data.columns
                         if column not in numeric_data.columns[qconstant_filter.get_support()]]
    print("quasi-constant features:", qconstant_columns)
    data = data.drop(columns=qconstant_columns + mydata_const_cols)
    # checking for duplicates
    data_T = data.T
    print(data_T.duplicated().sum())
    unique_features = data_T.drop_duplicates(keep='first').T
    # see the names of the duplicate columns
    duplicated_features = [dup_col for dup_col in data.columns if dup_col not in unique_features.columns]
    print("duplicated_features:", duplicated_features)
    data = data.drop(columns=duplicated_features)
    return data


def individual_t_test(data, alpha_val):
    '''
    For continuous variable individual t-tests
    '''

    # Separate the DataFrame based on the binary column
    group1 = data[data['TARGET'] == 0]
    group2 = data[data['TARGET'] == 1]

    # Initialize a list to store the results
    results = []

    # Iterate through each numeric column
    for col in data.columns:
        if col != 'binary_col' and np.issubdtype(data[col].dtype, np.number):
            # Perform t-test
            t_stat, p_value = ttest_ind(
                group1[col], group2[col], nan_policy='omit')
            if p_value < alpha_val:
                sig = 'Significant'
            else:
                sig = 'Insignificant'

            results.append({
                'column': col,
                't_stat': t_stat,
                'p_value': p_value,
                'significance': sig
            })

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results)

    print(results_df)
    return results_df


def perform_logistic_regression(data, alpha_val):
    # Create an empty list to store the results
    results = []

    # Loop over columns
    for col in data.loc[:, ~data.columns.isin(['customer_id', 'TARGET', 'loan_type'])].columns:
        X = data[[col]]
        X = sm.add_constant(X)
        y = data['TARGET']

        # Fit the logistic regression model
        model = sm.Logit(y, X).fit(disp=False)

        # Extract the p-values and z-statistics
        p_value = model.pvalues[col]
        z_stat = model.tvalues[col]

        if p_value < alpha_val:
            sig = 'Significant'
        else:
            sig = 'Insignificant'

        # Append the results to the list
        results.append({'Column': col, 'P-value': p_value, 'Z-stat': z_stat, 'Significance': sig})

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)

    print(results_df)
    return results_df


def save_data_for_binning(final_data):
    """
    Save the preprocessed data to a  CSV file for Binning.

    Args:
        final_data (pd.DataFrame): The clean data to be saved.

    Returns:
        None
    """
    try:
        # Get the current working directory
        current_dir = os.getcwd()
        subdirectory = "02.eda\\outputs"
        filename = "02.data.csv"

        # Combine the paths
        file_path = os.path.join(current_dir, subdirectory, filename)
        final_data.to_csv(path_or_buf=file_path, index=False)
        print("Data successfully saved for  Binning.")
    except Exception as e:
        print(f"Error writing model data for Binning: {e}")



def main():
    # Set the project root directory
    set_project_root()

    # Load eda data
    current_dir = os.getcwd()
    subdirectory = "01.data_processing\\outputs"
    subdirectory1 = "02.eda\\outputs"
    filename = "01.data.csv"
    file_path = os.path.join(current_dir, subdirectory, filename)
    data = load_data(file_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(os.path.dirname(os.path.join(current_dir, subdirectory))):
        os.makedirs(os.path.dirname(os.path.join(current_dir, subdirectory)))

    if data.empty:
        print("No EDA data available. Exiting...")
        return
    print(data.info())
    # Feature engineering
    logger.info("Starting transaction feature engineering...")
    try:
        data = generate_features(data)
        logger.success("Feature engineering completed.")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return

    logger.info("Shape after feature engineering: {}".format(data.shape))

    # Summary statistics for the variables
    feature_summary(data, subdirectory1)

    # Handle missing values 
    data = handle_missing_values(data, threshold=90, method = "drop")
    if data.empty:
        print("Error handling missing values in the data. Exiting...")
        return
    print(data.dtypes)

    # Handling Constant and Quasi-Constant
    final_data = handling_contant_and_quasi_contant(data)
    print("final_data_cols:", final_data.columns)
    print("final_shape:", final_data.shape)

    # Select only the desired columns, formatted for readability
    final_data = final_data[
        [
            "customer_id", "merchant_id",
            "days_since_last_txn", "days_since_first_txn", "txn_days_range",

            # 7-day features
            "txn_count_7d", "sum_amount_7d", "avg_amount_7d", "max_amount_7d",
            "min_amount_7d", "std_amount_7d", "cv_amount_7d",
            "mean_abs_z_7d", "max_abs_z_7d", "active_days_7d",
            "avg_txn_per_day_7d", "amount_min_max_ratio_7d",

            # 1-month features
            "txn_count_1m", "sum_amount_1m", "avg_amount_1m", "max_amount_1m",
            "min_amount_1m", "std_amount_1m", "cv_amount_1m",
            "mean_abs_z_1m", "max_abs_z_1m", "active_days_1m",
            "avg_txn_per_day_1m", "amount_min_max_ratio_1m",

            # 3-month features
            "txn_count_3m", "sum_amount_3m", "avg_amount_3m", "max_amount_3m",
            "min_amount_3m", "std_amount_3m", "cv_amount_3m",
            "mean_abs_z_3m", "max_abs_z_3m", "active_days_3m",
            "avg_txn_per_day_3m", "amount_min_max_ratio_3m",

            # 6-month features
            "txn_count_6m", "sum_amount_6m", "avg_amount_6m", "max_amount_6m",
            "min_amount_6m", "std_amount_6m", "cv_amount_6m",
            "mean_abs_z_6m", "max_abs_z_6m", "active_days_6m",
            "avg_txn_per_day_6m", "amount_min_max_ratio_6m",

            # Overall features
            "txn_count_overall", "sum_amount_overall", "avg_amount_overall",
            "max_amount_overall", "min_amount_overall", "std_amount_overall",
            "active_days_all", "cv_amount_overall", "avg_txn_per_day_overall",
            "mean_abs_z_overall", "max_abs_z_overall", "avg_days_between_txn",

            # Ratios and growth metrics
            "dependency_ratio_7d", "dependency_ratio_1m",
            "dependency_ratio_3m", "dependency_ratio_6m",
            "dependency_ratio_overall",
            "txn_count_growth_1m_vs_3m", "txn_count_growth_3m_vs_6m",
            "sum_amount_growth_1m_vs_3m", "sum_amount_growth_3m_vs_6m",
            "sum_amount_slope_6m", "txn_count_slope_6m",
            "rec_ratio_txn_7d_1m", "rec_ratio_txn_1m_3m",
            "txn_density_3m", "txn_density_6m",
            "vol_recency_3m", "vol_recency_6m",
            "avg_amt_ratio_1m_6m", "avg_amt_ratio_7d_3m",
            "txn_slope_7d_1m", "txn_slope_1m_3m",

            # Loan-related columns
            "loan_type", "loan_total_due", "loan_repaid_amounts",
            "loan_repayment_rate", "loan_repay_days", "loan_shortfall",

            # Target label
            "TARGET",
        ]
    ]


    # save EDA tables and plots
    distribution_plot(final_data, subdirectory1)
    # perform_logistic_regression(final_data, 0.05)
    plot_bivariate_distributions(final_data, final_data.loc[:,
                                             ~final_data.columns.isin(['customer_id', 'TARGET'])].columns.tolist(),
                                 final_data.TARGET, subdirectory1)
    bivariate_analysis(final_data, subdirectory1)
    individual_t_test(final_data, 0.05)
    # Save the preprocessed data for Binning
    save_data_for_binning(final_data)

    mis_val_table = missing_values_table(final_data)
    print(mis_val_table.to_string())
    print("final_data_missing", final_data.isnull().sum())

if __name__ == "__main__":
    main()

