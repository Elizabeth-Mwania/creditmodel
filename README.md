# Credit Scoring Model Developement

## Project Overview

This repository contains a complete pipeline for developing a credit scoring model consisting of statistical and machine learning methods. 
The model predicts loan repayment behavior using transaction-level data of a merchant lending firm. The pipeline addresses challenges like class imbalance and the need for interpretable, regulatory-compliant scoring to achieve decision making in credit scoring. 
Four mdoels are trained for comparison ie Logisti Rgression, Random Forest, Gradient Boosting and Decision Tree.

## Business Objectives

- **Risk Assessment**: Identify high-risk borrowers to minimize financial losses.
- **Lending Optimization**: Balance approval rates with default risk.
- **Interpretability**: Use scorecard-based binning for transparent decisions.
- **Scalability**: Modular pipeline for production deployment.

## Key Features

- **Data Pipeline**: From raw transactions to model-ready features.
- **EDA & Feature Engineering**: RFM metrics, temporal aggregations, log transformations.
- **Target Definition**: Binary classification (good/bad payers) based on repayment rate and timing.
- **Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
- **Evaluation**: ROC AUC, Gini, KS statistics, business scorecard analysis.
- **Reports**: Well documentated report including data description, preprocessing, model fitting and model evaluation.

## Project Structure

```
credit_score_model/
├── 00.data/
│   ├── initial_analysis.ipynb     # Initial understanding of the data
│   └── outputs/                   # Data for preprocessing
    └── raw-data
├── 01.data_processing/
│   ├── main.py                    # Target variable creation
│   └── outputs/                   # Preprocessed data for EDA
├── 02.eda/
│   ├── feature_engineering.py     # Feature engineering 
│   └── main.py                    # EDA execution
├── 03.binning_process/          # WoE transformation and feature selection
│   ├── main.py
│   ├── credit_binning_process.py
│   ├── binning_config.py
│   ├── binning_utilities.py
├── 04.model_fitting/            # Model training and hyperparameter tuning
│   ├── main.py
│   ├── model_fitting_engine.py
│   ├── model_config.py
│   └── outputs/
└── 05.model_evaluation/         # Model validation and business analysis
    ├── main.py
└── model_development_report.md  # Full project report
```

## Technologies Used

- **Python**: 3.10 with pandas, numpy, scikit-learn, matplotlib, seaborn
- **Jupyter Notebooks**: For exploratory analysis
- **Logging**: Loguru for pipeline monitoring
- **Visualization**: Plotly, Seaborn for insights

## Installation and Setup

1. **Clone the Repository**:
   ```
   git clone https://github.com/Elizabeth-Mwania/credit-score-model.git
   cd credit-score-model
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place `raw-data.csv` in `00.data/`
   - Run initial analysis: `jupyter notebook 00.data/initial_analysis.ipynb`

4. **Run the Pipeline**:
   - Data Processing
    ```
    cd 01.data_processing
    python main.py
    ```
   - EDA and Feature Engineering:
     ```
     cd ../02.eda
     python main.py
     ```
   - Binning and Feature Selection:
     ```
     cd ../03.binning_process
     python main.py
     ```
     - Model Training:
     ```
     cd ../04.model_fitting
     python main.py
     ```
     - Model Training:
     ```
     cd ../05.model_evaluation
     python main.py
     ```

## Pipeline Stages

### Stage 1: Data Processing (`01.data_processing/`)

**Purpose**: Transform raw transaction data and create binary target variable

**Key Functions**:
- `load_model_data()`: Loads and validates raw data
- `set_project_root()`: Establishes working directory
- Target creation logic: Good customers (TARGET=0) vs Bad customers (TARGET=1)

**Target Definition**:
```python
# Good customers: repayment_rate >= 100% AND repay_days <= 10
# Bad customers: All others
TARGET = (loan_repayment_rate >= 100) & (loan_repay_days <= 10)
TARGET = TARGET.replace({1: 0, 0: 1})  # Invert for modeling
```

**Output**: `01.data.csv` - Processed dataset with binary target

### Stage 2: Exploratory Data Analysis (`02.eda/`)

**Purpose**: Generate features and perform statistical analysis

**Core Components**:

#### Feature Engineering (`feature_engineering.py`)
- **Recency Features**: Days since last/first transaction, transaction span
- **Time-Windowed Features**: 7-day, 1-month, 3-month, 6-month transaction patterns
- **Statistical Features**: Mean, median, std dev, coefficient of variation, z-scores
- **Behavioral Features**: Transaction frequency, amount volatility, dependency ratios
- **Growth Metrics**: Period-over-period transaction growth, trend slopes
- **Interaction Features**: Cross-feature relationships and ratios

#### Key Feature Categories

| Category | Examples | Business Logic |
|----------|----------|----------------|
| **Recency** | `days_since_last_txn`, `txn_days_range` | Recent activity indicates engagement |
| **Frequency** | `txn_count_7d`, `active_days_1m` | Higher frequency suggests stability |
| **Monetary** | `sum_amount_3m`, `avg_amount_6m` | Spending patterns indicate capacity |
| **Volatility** | `cv_amount_1m`, `std_amount_3m` | Low volatility suggests predictability |
| **Growth** | `txn_count_growth_1m_vs_3m` | Positive trends indicate improvement |
| **Dependencies** | `dependency_ratio_7d` | High single-transaction dependency = risk |

#### Statistical Analysis
- **Missing Value Analysis**: Comprehensive missingness patterns
- **Univariate Analysis**: Distribution analysis and summary statistics  
- **Bivariate Analysis**: Feature-target relationships via t-tests and logistic regression
- **Constant/Quasi-constant Detection**: Removes non-informative features
- **Correlation Analysis**: Identifies multicollinearity issues

**Outputs**: 
- `02.data.csv`: Feature-engineered dataset
- `summary_statistics.csv`: Feature statistics
- Multiple visualization files for EDA insights

### Stage 3: Binning Process (`03.binning_process/`)

**Purpose**: Transform features using Weight of Evidence (WoE) and select optimal features

**Key Concepts**:

#### Weight of Evidence (WoE)
```
WoE = ln(Distribution of Goods / Distribution of Bads)
```
- Transforms continuous variables into discrete bins
- Maintains monotonic relationship with target
- Handles missing values and outliers automatically

#### Information Value (IV)
```
IV = Σ (% Goods - % Bads) * WoE
```
- Measures predictive power of each feature
- Selection criteria:
  - IV < 0.02: Not useful
  - 0.02 ≤ IV < 0.1: Weak predictor  
  - 0.1 ≤ IV < 0.3: Medium predictor
  - 0.3 ≤ IV < 0.5: Strong predictor
  - IV ≥ 0.5: Suspicious (potential overfitting)

**Configuration**: Customize binning parameters in `binning_config.py`
- Maximum bins per feature
- Monotonic trend requirements
- Minimum bin population
- IV thresholds for selection

**Outputs**:
- `03.train_data.csv` & `03.test_data.csv`: WoE-transformed datasets
- WOE plots and IV table
- Selected features based on IV and quality sscore

### Stage 4: Model Fitting (`04.model_fitting/`)

**Purpose**: Train multiple algorithms and select optimal model

**Supported Algorithms**:
- **Logistic Regression**: High interpretability, base model
- **Random Forest**: Handles non-linearity, ensemble method
- **Gradient Boosting**: High predictive power ensemble
- **Desicion Tree**: Simple non-linear model


**Key Features**:
- **Automated Hyperparameter Tuning**: Grid search with cross-validation
- **Class Imbalance Handling**: SMOTE, class weights, threshold optimization
- **Feature Scaling**: StandardScaler, RobustScaler options
- **Cross-Validation**: Stratified k-fold for robust performance estimation
- **Model Persistence**: Automated model and scaler serialization

**Configuration**: Customize training in `model_config.py`
```python
MODEL_TRAINING_CONFIG = {
    'cv_folds': 5,
    'scoring_metric': 'roc_auc',
    'hyperparameter_tuning': True,
    'random_state': 42
}
```

**Outputs**:
- Individual model files (`*_model.pkl`)
- Best model file (`best_model.pkl`)
- Feature importance analysis
- Mode evaluation report base on `Accuracy,Precision,Recall,F1_Score,ROC_AUC,PR_AUC,Gini,KS_Statistic,Log_Loss` and model summary report

### Stage 5: Model Evaluation (`05.model_evaluation/`)

**Purpose**: Performs model validation and business analysis

**Evaluation Metrics**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **ROC AUC** | Area Under ROC Curve | Overall discrimination ability |
| **Gini Coefficient** | 2 * AUC - 1 | Normalized discrimination measure |
| **KS Statistic** | max(TPR - FPR) | Maximum separation between classes |
| **Precision** | TP / (TP + FP) | Accuracy of positive predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual positives |

**Performance Thresholds**:
- **Excellent**: AUC ≥ 0.80, Gini ≥ 0.60, KS ≥ 0.40
- **Good**: AUC ≥ 0.70, Gini ≥ 0.40, KS ≥ 0.30  
- **Acceptable**: AUC ≥ 0.60, Gini ≥ 0.20, KS ≥ 0.20

**Business Analysis**:
- **Scorecard Development**: 10-band risk segmentation
- **Population Stability**: Score distribution analysis
- **Lift Analysis**: Model effectiveness across risk bands
- **Profitability Assessment**: Business impact quantification

**Outputs**:
- Evaluation reports
- Business scorecard analysis
- Model comparison visualizations

## Key Performance Indicators

Monitor these metrics throughout the pipeline:

### Data Quality
- Missing value percentage < 5%
- Constant feature removal count
- Feature correlation levels

### Feature Engineering  
- Information Value distribution
- Feature stability across time periods
- WoE monotonicity compliance

### Model Performance
- Cross-validation stability (std < 0.02)
- Train-test performance gap < 0.05
- Business metric alignment

## Customization Guide

### Adding New Features
1. Extend `feature_engineering.py` with new feature functions
2. Update feature lists in subsequent stages
3. Configure binning parameters if needed

### Integrating New Algorithms
1. Add algorithm to `model_config.py`
2. Define hyperparameter grids
3. Update evaluation framework

### Modifying Business Rules
1. Update target definition in `01.data_processing/main.py`
2. Adjust performance thresholds in `model_config.py`
3. Modify scorecard bands in evaluation stage

## Common Issues & Troubleshooting

### Data Issues
**Problem**: High missing value rates
**Solution**: Implement imputation strategies in `feature_engineering.py`

**Problem**: Insufficient historical data
**Solution**: Adjust time windows in feature engineering configuration

### Model Issues  
**Problem**: Poor model performance (AUC < 0.60)
**Solutions**:
- Review feature engineering logic
- Increase data collection period
- Consider external data sources

**Problem**: Model overfitting (train-test gap > 0.10)
**Solutions**:
- Increase regularization parameters
- Reduce feature dimensionality
- Implement more aggressive cross-validation

### Technical Issues
**Problem**: Memory errors during processing
**Solutions**:
- Implement data chunking
- Reduce feature set size
- Use more efficient data types

## Model Deployment Checklist

- [ ] Model AUC ≥ 0.70 (minimum acceptable)
- [ ] Gini coefficient ≥ 0.40
- [ ] KS statistic ≥ 0.30  
- [ ] Population stability index < 0.10
- [ ] Feature stability validation complete
- [ ] Business scorecard validated
- [ ] Documentation complete
- [ ] Regulatory compliance verified

## References & Further Reading

1. **Credit Risk Analytics**: Measurement Techniques, Applications, and Examples in SAS
2. **Developing Credit Risk Models Using SAS Enterprise Miner and SAS/STAT**  
3. **The Credit Scoring Toolkit**: Theory and Practice for Retail Credit Risk Management
4. **Basel II/III Guidelines**: Risk Management and Capital Adequacy

## Contributing

Contributions welcome! Please follow PEP 8 standards and add tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, reach out via GitHub issues.

---

*Note: Raw data is sensitive; this repo includes synthetic/processed examples. Full implementation requires secure data handling.*
