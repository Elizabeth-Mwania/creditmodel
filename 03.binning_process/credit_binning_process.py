import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import warnings
import os
from loguru import logger
from typing import Dict, List, Tuple, Optional, Union
import json

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class CreditScoringBinning:
    """
    Credit scoring binning class that handles:
    - Information Value (IV) calculation
    - Weight of Evidence (WoE) calculation
    - Optimal binning using various methods
    - Variable selection based on IV and quality metrics
    - Visualization of binning results
    """
    
    def __init__(self, name: str, dtype: str = 'numerical', max_bins: int = 10, 
                 min_bin_size: float = 0.05, monotonic_trend: str = None,
                 user_splits: List = None):
        """
        Initialize the binning object
        
        Parameters:
        - name: Variable name
        - dtype: 'numerical' or 'categorical'
        - max_bins: Maximum number of bins
        - min_bin_size: Minimum percentage of observations per bin
        - monotonic_trend: 'ascending', 'descending', or None
        - user_splits: Predefined split points for binning
        """
        self.name = name
        self.dtype = dtype
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.monotonic_trend = monotonic_trend
        self.user_splits = user_splits
        self.binning_table = None
        self.woe_dict = {}
        self.iv = 0
        self.is_fitted = False
        
    def calculate_woe_iv(self, df_grouped: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """Calculate Weight of Evidence and Information Value"""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        
        # Calculate WoE and IV
        df_grouped['good_rate'] = df_grouped['good'] / (df_grouped['good'] + df_grouped['bad'] + epsilon)
        df_grouped['bad_rate'] = df_grouped['bad'] / (df_grouped['good'] + df_grouped['bad'] + epsilon)
        
        total_good = df_grouped['good'].sum()
        total_bad = df_grouped['bad'].sum()
        
        df_grouped['good_dist'] = df_grouped['good'] / (total_good + epsilon)
        df_grouped['bad_dist'] = df_grouped['bad'] / (total_bad + epsilon)
        
        # Calculate WoE with epsilon to avoid log(0)
        df_grouped['WoE'] = np.log((df_grouped['good_dist'] + epsilon) / (df_grouped['bad_dist'] + epsilon))
        df_grouped['IV_component'] = (df_grouped['good_dist'] - df_grouped['bad_dist']) * df_grouped['WoE']
        
        total_iv = df_grouped['IV_component'].sum()
        
        return total_iv, df_grouped
    
    def optimal_binning_decision_tree(self, x: np.array, y: np.array) -> List:
        """Use decision tree for optimal binning"""
        if len(np.unique(x)) <= 2:
            return sorted(np.unique(x))
            
        dt = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=int(len(x) * self.min_bin_size),
            random_state=42
        )
        
        x_reshaped = x.reshape(-1, 1)
        dt.fit(x_reshaped, y)
        
        # Extract thresholds from decision tree
        thresholds = []
        tree = dt.tree_
        
        def extract_thresholds(node_id):
            if tree.children_left[node_id] != tree.children_right[node_id]:  # Not a leaf
                thresholds.append(tree.threshold[node_id])
                extract_thresholds(tree.children_left[node_id])
                extract_thresholds(tree.children_right[node_id])
        
        extract_thresholds(0)
        thresholds = sorted(list(set(thresholds)))
        
        # Add min and max values
        splits = [x.min()] + thresholds + [x.max()]
        return sorted(list(set(splits)))
    
    def equal_frequency_binning(self, x: np.array) -> List:
        """Create equal frequency bins"""
        try:
            quantiles = np.linspace(0, 1, self.max_bins + 1)
            splits = np.quantile(x, quantiles)
            return sorted(list(set(splits)))
        except:
            return [x.min(), x.max()]
    
    def create_bins(self, x: np.array, y: np.array) -> List:
        """Create optimal bins based on the specified method"""
        if self.user_splits:
            return self.user_splits
        
        if self.dtype == 'numerical':
            # Try decision tree binning first
            try:
                splits = self.optimal_binning_decision_tree(x, y)
                if len(splits) > 2:
                    return splits
            except:
                pass
            
            # Fallback to equal frequency
            return self.equal_frequency_binning(x)
        
        else:  # categorical
            return sorted(list(set(x)))
    
    def enforce_monotonicity(self, binning_table: pd.DataFrame) -> pd.DataFrame:
        """Enforce monotonic trend in WoE values"""
        if self.monotonic_trend is None:
            return binning_table
        
        # Sort by bin order
        binning_table = binning_table.sort_index()
        woe_values = binning_table['WoE'].values
        
        if self.monotonic_trend == 'ascending':
            # Ensure WoE is non-decreasing
            for i in range(1, len(woe_values)):
                if woe_values[i] < woe_values[i-1]:
                    woe_values[i] = woe_values[i-1]
        
        elif self.monotonic_trend == 'descending':
            # Ensure WoE is non-increasing
            for i in range(1, len(woe_values)):
                if woe_values[i] > woe_values[i-1]:
                    woe_values[i] = woe_values[i-1]
        
        binning_table['WoE'] = woe_values
        
        # Recalculate IV components
        binning_table['IV_component'] = (binning_table['good_dist'] - binning_table['bad_dist']) * binning_table['WoE']
        
        return binning_table
    
    def fit(self, x: np.array, y: np.array):
        """Fit the binning to the data"""
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        
        # Remove missing values
        mask = ~(pd.isna(x) | pd.isna(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) == 0:
            raise ValueError("No valid data points after removing missing values")
        
        # Create bins
        if self.dtype == 'numerical':
            splits = self.create_bins(x_clean, y_clean)
            
            # Create bin labels
            df = pd.DataFrame({'x': x_clean, 'y': y_clean})
            df['bin'] = pd.cut(df['x'], bins=splits, include_lowest=True, duplicates='drop')
            
        else:  # categorical
            df = pd.DataFrame({'x': x_clean, 'y': y_clean})
            df['bin'] = df['x']
        
        # Group by bins and calculate statistics
        grouped = df.groupby('bin').agg({
            'y': ['count', 'sum']
        }).round(4)
        
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        
        # Remove bins that are too small
        min_count = int(len(df) * self.min_bin_size)
        grouped = grouped[grouped['total'] >= min_count]
        
        if len(grouped) == 0:
            raise ValueError(f"No bins meet the minimum size requirement of {self.min_bin_size}")
        
        # Calculate WoE and IV
        self.iv, self.binning_table = self.calculate_woe_iv(grouped)
        
        # Enforce monotonicity if specified
        if self.monotonic_trend:
            self.binning_table = self.enforce_monotonicity(self.binning_table)
            # Recalculate IV after monotonicity enforcement
            self.iv = self.binning_table['IV_component'].sum()
        
        # Create WoE mapping dictionary
        self.woe_dict = self.binning_table['WoE'].to_dict()
        
        self.is_fitted = True
        
        return self
    
    def transform(self, x: np.array) -> np.array:
        """Transform data using fitted WoE values"""
        if not self.is_fitted:
            raise ValueError("Binning must be fitted before transform")
        
        if self.dtype == 'numerical':
            # Find which bin each value belongs to
            woe_values = []
            for val in x:
                if pd.isna(val):
                    woe_values.append(0)  # Default WoE for missing values
                    continue
                
                assigned = False
                for bin_interval, woe in self.woe_dict.items():
                    if pd.Interval.left <= val <= pd.Interval.right:
                        woe_values.append(woe)
                        assigned = True
                        break
                
                if not assigned:
                    # If value doesn't fall in any bin, assign closest bin's WoE
                    woe_values.append(list(self.woe_dict.values())[0])
            
            return np.array(woe_values)
        
        else:  # categorical
            return np.array([self.woe_dict.get(val, 0) for val in x])
    
    def plot_woe(self, save_path: str = None, figsize: Tuple[int, int] = (10, 6)):
        """Plot WoE values by bins"""
        if not self.is_fitted:
            raise ValueError("Binning must be fitted before plotting")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # WoE by bins
        bins = range(len(self.binning_table))
        woe_values = self.binning_table['WoE'].values
        
        ax1.bar(bins, woe_values, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Bins')
        ax1.set_ylabel('Weight of Evidence')
        ax1.set_title(f'WoE Plot for {self.name}')
        ax1.grid(True, alpha=0.3)
        
        # Add bin labels
        bin_labels = [str(idx) for idx in self.binning_table.index]
        ax1.set_xticks(bins)
        ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
        
        # Good/Bad rate by bins
        good_rate = self.binning_table['good_rate'].values
        bad_rate = self.binning_table['bad_rate'].values
        
        x = np.arange(len(bins))
        width = 0.35
        
        ax2.bar(x - width/2, good_rate, width, label='Good Rate', alpha=0.7, color='green')
        ax2.bar(x + width/2, bad_rate, width, label='Bad Rate', alpha=0.7, color='red')
        
        ax2.set_xlabel('Bins')
        ax2.set_ylabel('Rate')
        ax2.set_title(f'Good/Bad Rates by Bins for {self.name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def get_binning_summary(self) -> pd.DataFrame:
        """Get binning summary table"""
        if not self.is_fitted:
            raise ValueError("Binning must be fitted before getting summary")
        
        summary = self.binning_table.copy()
        summary['Variable'] = self.name
        summary['Bin'] = summary.index
        summary['IV'] = self.iv
        
        # Reorder columns
        cols = ['Variable', 'Bin', 'total', 'good', 'bad', 'good_rate', 'bad_rate', 'WoE', 'IV_component', 'IV']
        return summary[cols]


class CreditScoringBinningProcess:
    """
    Main class for handling multiple variables binning process
    """
    
    def __init__(self, variable_names: List[str], selection_criteria: Dict = None, 
                 min_bin_size: float = 0.05, binning_params: Dict = None):
        """
        Initialize the binning process
        
        Parameters:
        - variable_names: List of variable names to process
        - selection_criteria: Criteria for variable selection (IV, quality score, etc.)
        - min_bin_size: Minimum bin size as proportion
        - binning_params: Dictionary of binning parameters for each variable
        """
        self.variable_names = variable_names
        self.selection_criteria = selection_criteria or {
            "iv": {"min": 0.02, "max": 0.8},
            "quality_score": {"min": 0.01}
        }
        self.min_bin_size = min_bin_size
        self.binning_params = binning_params or {}
        self.fitted_binnings = {}
        self.summary_table = None
        self.is_fitted = False
    
    def calculate_quality_score(self, binning_table: pd.DataFrame) -> float:
        """Calculate quality score based on various criteria"""
        # Basic quality score based on:
        # 1. Number of bins (penalize too few or too many)
        # 2. Monotonicity (if trend is specified)
        # 3. Bin size distribution
        
        num_bins = len(binning_table)
        
        # Penalize too few bins (< 3) or too many bins (> 10)
        if num_bins < 3:
            bin_penalty = 0.5
        elif num_bins > 10:
            bin_penalty = 0.7
        else:
            bin_penalty = 1.0
        
        # Check for empty or very small bins
        min_bin_size_penalty = 1.0
        total_count = binning_table['total'].sum()
        for _, row in binning_table.iterrows():
            if row['total'] / total_count < self.min_bin_size:
                min_bin_size_penalty *= 0.8
        
        # Base quality score
        quality_score = bin_penalty * min_bin_size_penalty * 0.5
        
        return min(max(quality_score, 0.01), 1.0)  # Bound between 0.01 and 1.0
    
    def fit(self, X: pd.DataFrame, y: np.array):
        """Fit binning for all variables"""
        results = []
        
        for var_name in self.variable_names:
            if var_name not in X.columns:
                logger.warning(f"Variable {var_name} not found in dataset")
                continue
            
            try:
                # Get binning parameters for this variable
                params = self.binning_params.get(var_name, {})
                
                # Create binning object
                binning = CreditScoringBinning(
                    name=var_name,
                    dtype=params.get('dtype', 'numerical'),
                    max_bins=params.get('max_bins', 10),
                    min_bin_size=self.min_bin_size,
                    monotonic_trend=params.get('monotonic_trend'),
                    user_splits=params.get('user_splits')
                )
                
                # Fit binning
                x_values = X[var_name].values
                binning.fit(x_values, y)
                
                # Calculate quality score
                quality_score = self.calculate_quality_score(binning.binning_table)
                
                # Check selection criteria
                selected = True
                iv_criteria = self.selection_criteria.get('iv', {})
                if 'min' in iv_criteria and binning.iv < iv_criteria['min']:
                    selected = False
                if 'max' in iv_criteria and binning.iv > iv_criteria['max']:
                    selected = False
                
                quality_criteria = self.selection_criteria.get('quality_score', {})
                if 'min' in quality_criteria and quality_score < quality_criteria['min']:
                    selected = False
                
                # Store results
                self.fitted_binnings[var_name] = binning
                
                results.append({
                    'name': var_name,
                    'dtype': params.get('dtype', 'numerical'),
                    'bins': len(binning.binning_table),
                    'iv': binning.iv,
                    'quality_score': quality_score,
                    'selected': selected,
                    'monotonic_trend': params.get('monotonic_trend', 'None')
                })
                
                logger.info(f"Successfully fitted binning for {var_name}: IV={binning.iv:.4f}, Quality={quality_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error fitting binning for {var_name}: {str(e)}")
                results.append({
                    'name': var_name,
                    'dtype': 'unknown',
                    'bins': 0,
                    'iv': 0,
                    'quality_score': 0,
                    'selected': False,
                    'monotonic_trend': 'None',
                    'error': str(e)
                })
        
        self.summary_table = pd.DataFrame(results)
        self.is_fitted = True
        
        return self
    
    def summary(self) -> pd.DataFrame:
        """Get summary of all binning results"""
        if not self.is_fitted:
            raise ValueError("Must fit binning process first")
        
        return self.summary_table.copy()
    
    def get_selected_variables(self) -> List[str]:
        """Get list of selected variables"""
        if not self.is_fitted:
            raise ValueError("Must fit binning process first")
        
        return self.summary_table[self.summary_table['selected']]['name'].tolist()
    
    def transform(self, X: pd.DataFrame, selected_only: bool = True) -> pd.DataFrame:
        """Transform variables using fitted binnings"""
        if not self.is_fitted:
            raise ValueError("Must fit binning process first")
        
        variables_to_transform = self.get_selected_variables() if selected_only else list(self.fitted_binnings.keys())
        
        X_transformed = X.copy()
        
        for var_name in variables_to_transform:
            if var_name in self.fitted_binnings:
                binning = self.fitted_binnings[var_name]
                X_transformed[f"{var_name}_WoE"] = binning.transform(X[var_name].values)
        
        return X_transformed
    
    def save_binning_tables(self, output_folder: str):
        """Save individual binning tables for each variable"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for var_name, binning in self.fitted_binnings.items():
            summary = binning.get_binning_summary()
            file_path = os.path.join(output_folder, f"{var_name}_binning_table.csv")
            summary.to_csv(file_path, index=False)
            logger.info(f"Binning table saved for {var_name}")


def create_directory_if_not_exists(path: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory created: {path}")


def woe_iv_filter(train: pd.DataFrame, test: pd.DataFrame, 
                 binning_params: Dict, output_folder: str = "output") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter dataset based on IV and quality scores using business-ready binning process
    """
    create_directory_if_not_exists(output_folder)
    
    try:
        # Define selection criteria
        selection_criteria = {
            "iv": {"min": 0.02, "max": 0.8},
            "quality_score": {"min": 0.01}
        }
        
        # Get variable names (excluding customer_id and TARGET)
        variable_names = [col for col in train.columns if col not in ['customer_id','merchant_id', 'TARGET']]
        
        # Initialize binning process
        binning_process = CreditScoringBinningProcess(
            variable_names=variable_names,
            selection_criteria=selection_criteria,
            min_bin_size=0.05,
            binning_params=binning_params
        )
        
        # Prepare data
        X = train[variable_names]
        y = train["TARGET"].values
        
        # Fit binning process
        binning_process.fit(X, y)
        
        # Get summary
        summary_tab = binning_process.summary()
        print("Binning Summary:")
        print(summary_tab.to_string(index=False))
        
        # Save IV table
        iv_table_path = os.path.join(output_folder, "iv_table.csv")
        summary_tab.to_csv(iv_table_path, index=False)
        logger.info(f"IV table saved to {iv_table_path}")
        
        # Save individual binning tables
        binning_process.save_binning_tables(output_folder)
        
        # Filter variables based on selection criteria
        selected_vars = binning_process.get_selected_variables()
        selected_vars_with_ids = selected_vars + ['customer_id', 'TARGET']
        
        train_filtered = train[selected_vars_with_ids]
        test_filtered = test[selected_vars_with_ids]
        
        print(f"Selected variables: {selected_vars}")
        print(f"Total variables selected: {len(selected_vars)}")
        
        return train_filtered, test_filtered, binning_process
        
    except Exception as e:
        logger.error(f"Error in woe_iv_filter: {e}")
        raise


def generate_woe_plots(train: pd.DataFrame, binning_process: CreditScoringBinningProcess, 
                      output_folder: str):
    """Generate WoE plots for all fitted variables"""
    create_directory_if_not_exists(output_folder)
    
    for var_name, binning in binning_process.fitted_binnings.items():
        try:
            plot_path = os.path.join(output_folder, f"{var_name}_woe_plot.png")
            binning.plot_woe(save_path=plot_path)
            logger.info(f"WoE plot saved for {var_name}")
        except Exception as e:
            logger.error(f"Error generating plot for {var_name}: {e}")


def define_binning_params() -> Dict:
    """Define comprehensive binning parameters for credit scoring variables"""
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
        "txn_days_range": {
            "dtype": "numerical",
            "monotonic_trend": "ascending",
            "max_bins": 6
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
        "min_amount_6m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "cv_amount_1m": {
            "dtype": "numerical",
            "monotonic_trend": "ascending",
            "max_bins": 6
        },
        "mean_abs_z_1m": {
            "dtype": "numerical",
            "monotonic_trend": "ascending",
            "max_bins": 6
        },
        "max_abs_z_3m": {
            "dtype": "numerical",
            "monotonic_trend": "ascending",
            "max_bins": 6
        },
        "avg_txn_per_day_1m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 8
        },
        "amount_min_max_ratio_3m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "txn_count_growth_1m_vs_3m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "sum_amount_growth_1m_vs_3m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "dependency_ratio_6m": {
            "dtype": "numerical",
            "monotonic_trend": "ascending",
            "max_bins": 6
        },
        "sum_amount_slope_6m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "txn_count_slope_6m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "rec_ratio_txn_1m_3m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "txn_density_3m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "vol_recency_3m": {
            "dtype": "numerical",
            "monotonic_trend": "ascending",
            "max_bins": 6
        },
        "avg_amt_ratio_1m_6m": {
            "dtype": "numerical",
            "monotonic_trend": "descending",
            "max_bins": 6
        },
        "txn_slope_7d_1m": {
            "dtype": "numerical",
            "monotonic_trend": "ascending",
            "max_bins": 6
        },
        "loan_type": {
            "dtype": "categorical",
            "max_bins": 5
        }
    }
