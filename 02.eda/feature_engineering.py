# feature_engineering.py
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.linear_model import LinearRegression


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generating features for credit scoring based on transaction history.

    """

    data = data.copy()

    # Ensure proper datetime conversion
    data['first_funded_date'] = pd.to_datetime(data['first_funded_date'], errors='coerce')
    data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')

    # Calculate days_before_loan

    data['days_before_loan'] = (data['first_funded_date'] - data['transaction_date']).dt.days

    # -------------------------------
    # Recency features
    # -------------------------------
    recency = data.groupby(['customer_id', 'merchant_id']).agg(
        days_since_last_txn=('days_before_loan', 'min'),
        days_since_first_txn=('days_before_loan', 'max'),
        txn_days_range=('days_before_loan', lambda x: x.max() - x.min())
    ).reset_index()

    # create window label for time-windowed features
    window_labels = {7: "7d", 30: "1m", 90: "3m", 180: "6m"}

    # -------------------------------
    # Transaction features by window
    # -------------------------------
    def generate_txn_features(df, windows=[7, 30, 90, 180]): 
        feature_dfs = []
        for w in windows:
            recent = df[df['days_before_loan'] <= w]
            label = window_labels[w]

            agg = recent.groupby('customer_id').agg(
                **{
                    f"txn_count_{label}": ('transaction_id', 'count'),
                    f"sum_amount_{label}": ('face_amount', 'sum'),
                    f"avg_amount_{label}": ('face_amount', 'mean'),
                    f"max_amount_{label}": ('face_amount', 'max'),
                    f"min_amount_{label}": ('face_amount', 'min'),
                    f"std_amount_{label}": ('face_amount', 'std'),
                }
            ).reset_index()

            # CV
            agg[f"cv_amount_{label}"] = agg[f"std_amount_{label}"] / agg[f"avg_amount_{label}"]
            agg[f"cv_amount_{label}"] = agg[f"cv_amount_{label}"].replace([np.inf, -np.inf], 0).fillna(0)

            # Z-scores
            if not recent.empty:
                recent = recent.copy()
                recent['z_score'] = recent.groupby('customer_id')['face_amount'] \
                    .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1))

                z_agg = recent.groupby('customer_id').agg(
                    mean_abs_z=('z_score', lambda x: x.abs().mean()),
                    max_abs_z=('z_score', lambda x: x.abs().max())
                ).reset_index()

                z_agg = z_agg.rename(columns={
                    "mean_abs_z": f"mean_abs_z_{label}",
                    "max_abs_z": f"max_abs_z_{label}"
                })
                agg = pd.merge(agg, z_agg, on='customer_id', how='left')

            # Active days
            active_days = recent.groupby('customer_id')['transaction_date'].nunique().reset_index()
            active_days = active_days.rename(columns={'transaction_date': f"active_days_{label}"})
            agg = pd.merge(agg, active_days, on='customer_id', how='left')

            # Ratios
            agg[f"active_days_ratio_{label}"] = agg[f"active_days_{label}"] / w
            agg[f"avg_txn_per_day_{label}"] = agg[f"txn_count_{label}"] / w
            agg[f"amount_min_max_ratio_{label}"] = agg[f"min_amount_{label}"] / agg[f"max_amount_{label}"]
            agg[f"amount_min_max_ratio_{label}"] = agg[f"amount_min_max_ratio_{label}"].replace([np.inf, -np.inf], 0).fillna(0)

            feature_dfs.append(agg.fillna(0))

        features = reduce(lambda l, r: pd.merge(l, r, on='customer_id', how='outer'), feature_dfs)
        return features.fillna(0)

    txn_features = generate_txn_features(data)

    # -------------------------------
    # Overall historical features
    # -------------------------------
    overall_agg = data.groupby('customer_id').agg(
        txn_count_overall=('transaction_id', 'count'),
        sum_amount_overall=('face_amount', 'sum'),
        avg_amount_overall=('face_amount', 'mean'),
        max_amount_overall=('face_amount', 'max'),
        min_amount_overall=('face_amount', 'min'),
        std_amount_overall=('face_amount', 'std'),
        active_days_all=('transaction_date', 'nunique'),
    ).reset_index()

    overall_agg['cv_amount_overall'] = overall_agg['std_amount_overall'] / overall_agg['avg_amount_overall']
    overall_agg['cv_amount_overall'] = overall_agg['cv_amount_overall'].replace([np.inf, -np.inf], 0).fillna(0)

    span_days = data.groupby('customer_id')['days_before_loan'].max().reset_index()
    span_days = span_days.rename(columns={'days_before_loan': 'span_days_all'})
    overall_agg = overall_agg.merge(span_days, on='customer_id', how='left')
    overall_agg['avg_txn_per_day_overall'] = overall_agg['txn_count_overall'] / overall_agg['span_days_all'].replace(0, 1)

    # Z-scores (all history)
    if not data.empty:
        data = data.copy()
        data['z_score'] = data.groupby('customer_id')['face_amount'] \
            .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1))
        z_agg_all = data.groupby('customer_id').agg(
            mean_abs_z_overall=('z_score', lambda x: x.abs().mean()),
            max_abs_z_overall=('z_score', lambda x: x.abs().max())
        ).reset_index()
        overall_agg = pd.merge(overall_agg, z_agg_all, on='customer_id', how='left')

    overall_agg = overall_agg.fillna(0)

    # -------------------------------
    # Avg days between transactions
    # -------------------------------
    data = data.sort_values(['customer_id', 'transaction_date'])
    data['days_between_txn'] = data.groupby('customer_id')['transaction_date'].diff().dt.days
    avg_gap = data.groupby('customer_id').agg(
        avg_days_between_txn=('days_between_txn', 'mean')
    ).reset_index().fillna(0)

    # -------------------------------
    # Merge core features
    # -------------------------------
    df_features = recency.merge(txn_features, on="customer_id", how="left").fillna(0)
    df_features = df_features.merge(overall_agg, on="customer_id", how="left").fillna(0)
    df_features = df_features.merge(avg_gap, on="customer_id", how="left").fillna(0)

    # -------------------------------
    # Dependency ratios
    # -------------------------------
    for w, label in window_labels.items():
        df_features[f'dependency_ratio_{label}'] = (
            df_features[f"max_amount_{label}"] / df_features[f"sum_amount_{label}"].replace(0, np.nan)
        )
    df_features['dependency_ratio_overall'] = (
        df_features['max_amount_overall'] / df_features['sum_amount_overall'].replace(0, np.nan)
    )
    df_features = df_features.fillna(0)

    # -------------------------------
    # Momentum & Growth Features
    # -------------------------------
    eps = 1e-9
    # growth
    df_features['txn_count_growth_1m_vs_3m'] = (df_features['txn_count_1m'] - df_features['txn_count_3m']) / (df_features['txn_count_3m'] + eps)
    df_features['txn_count_growth_3m_vs_6m'] = (df_features['txn_count_3m'] - df_features['txn_count_6m']) / (df_features['txn_count_6m'] + eps)
    df_features['sum_amount_growth_1m_vs_3m'] = (df_features['sum_amount_1m'] - df_features['sum_amount_3m']) / (df_features['sum_amount_3m'] + eps)
    df_features['sum_amount_growth_3m_vs_6m'] = (df_features['sum_amount_3m'] - df_features['sum_amount_6m']) / (df_features['sum_amount_6m'] + eps)

    for col in ['txn_count_growth_1m_vs_3m','txn_count_growth_3m_vs_6m','sum_amount_growth_1m_vs_3m','sum_amount_growth_3m_vs_6m']:
        df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # slope (6m trend)
    # -------------------------------
    # slope (6m trend)
    # -------------------------------
    recent_6m = data.loc[data['days_before_loan'] <= 180, 
                        ['customer_id','face_amount','transaction_id','days_before_loan']].copy()

    if not recent_6m.empty:
        # Create month index (0=oldest .. 5=most recent)
        recent_6m['month_order'] = 5 - (recent_6m['days_before_loan'] // 30).astype(int)

        # Pivot amounts
        month_amt = recent_6m.groupby(['customer_id','month_order'])['face_amount'].sum().reset_index()
        amt_pivot = (
            month_amt.pivot(index='customer_id', columns='month_order', values='face_amount')
            .reindex(columns=range(6), fill_value=0)
            .reset_index()
        )

        # Pivot counts
        month_cnt = recent_6m.groupby(['customer_id','month_order'])['transaction_id'].count().reset_index()
        cnt_pivot = (
            month_cnt.pivot(index='customer_id', columns='month_order', values='transaction_id')
            .reindex(columns=range(6), fill_value=0)
            .reset_index()
        )

        # X axis for regression
        X = np.arange(6).reshape(-1, 1)

        def slope_of_row(arr):
            # Ensure numeric
            arr = pd.to_numeric(arr, errors='coerce').fillna(0).values
            if np.all(arr == 0):
                return 0.0
            try:
                lr = LinearRegression()
                lr.fit(X, arr)
                return float(lr.coef_[0])
            except Exception:
                return 0.0

        # Compute slopes
        amt_slope = amt_pivot.drop(columns='customer_id').apply(slope_of_row, axis=1) \
            .rename('sum_amount_slope_6m')
        cnt_slope = cnt_pivot.drop(columns='customer_id').apply(slope_of_row, axis=1) \
            .rename('txn_count_slope_6m')

        amt_slope = pd.concat([amt_pivot['customer_id'], amt_slope], axis=1)
        cnt_slope = pd.concat([cnt_pivot['customer_id'], cnt_slope], axis=1)

        # Merge back
        df_features = df_features.merge(amt_slope, on='customer_id', how='left')
        df_features = df_features.merge(cnt_slope, on='customer_id', how='left')

    df_features = df_features.fillna(0)


    # -------------------------------
    # Interaction features
    # -------------------------------
    def generate_interactions(df):
        df_int = df.copy()
        df_int['rec_ratio_txn_7d_1m'] = df_int['txn_count_7d'] / df_int['txn_count_1m'].replace(0, np.nan)
        df_int['rec_ratio_txn_1m_3m'] = df_int['txn_count_1m'] / df_int['txn_count_3m'].replace(0, np.nan)
        df_int['txn_density_3m'] = df_int['txn_count_3m'] / df_int['span_days_all'].replace(0, 1)
        df_int['txn_density_6m'] = df_int['txn_count_6m'] / df_int['span_days_all'].replace(0, 1)
        if 'cv_amount_3m' in df_int:
            df_int['vol_recency_3m'] = df_int['cv_amount_3m'] * df_int['avg_days_between_txn']
        if 'cv_amount_6m' in df_int:
            df_int['vol_recency_6m'] = df_int['cv_amount_6m'] * df_int['avg_days_between_txn']
        df_int['avg_amt_ratio_1m_6m'] = df_int['avg_amount_1m'] / df_int['avg_amount_6m'].replace(0, np.nan)
        df_int['avg_amt_ratio_7d_3m'] = df_int['avg_amount_7d'] / df_int['avg_amount_3m'].replace(0, np.nan)
        df_int['txn_slope_7d_1m'] = df_int['txn_count_7d'] - (df_int['txn_count_1m'] / 4.0)
        df_int['txn_slope_1m_3m'] = df_int['txn_count_1m'] - (df_int['txn_count_3m'] / 3.0)
        return df_int.fillna(0)

    df_features = generate_interactions(df_features)

    # -------------------------------
    # Repayment info
    # -------------------------------
    repay_info = data[
        ['customer_id','loan_type','loan_total_due','loan_repaid_amounts',
         'loan_repayment_rate','loan_repay_days','loan_shortfall','TARGET']
    ].drop_duplicates()

    df_final = df_features.merge(repay_info, on='customer_id', how='left').fillna(0)
    return df_final
