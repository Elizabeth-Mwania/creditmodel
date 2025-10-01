"""
Scorecard utilities for credit scoring models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class CreditScorecard:
    """
    Credit scorecard class for converting logistic regression to interpretable scorecard
    """

    def __init__(self, base_score: int = 600, pdo: int = 20, odds: float = 20):
        """
        Initialize scorecard parameters

        Parameters:
        - base_score: Base score (default 600)
        - pdo: Points to Double the Odds (default 20)
        - odds: Odds at base score (default 20:1)
        """
        self.base_score = base_score
        self.pdo = pdo
        self.odds = odds

        # Calculate scorecard factors
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(odds)

        self.scorecard_table = {}
        self.feature_contributions = {}
        self.is_fitted = False

    def fit(self, model, binning_process, feature_names: List[str]):
        """
        Fit scorecard using trained logistic regression model and binning information

        Parameters:
        - model: Trained logistic regression model
        - binning_process: Fitted CreditScoringBinningProcess object
        - feature_names: List of feature names used in model
        """
        try:
            # Get model coefficients and intercept
            coef = model.coef_[0]
            intercept = model.intercept_[0]

            logger.info(f"Model intercept: {intercept}")
            logger.info(f"Number of coefficients: {len(coef)}")

            # Calculate base points from intercept
            base_points = self.offset + self.factor * intercept
            self.scorecard_table['base_points'] = round(base_points)

            logger.info(f"Base points calculated: {base_points}")

            # Calculate points for each feature
            total_features = 0
            for i, feature_name in enumerate(feature_names):
                if feature_name in binning_process.fitted_binnings:
                    binning = binning_process.fitted_binnings[feature_name]
                    feature_coef = coef[i]

                    # Calculate points for each bin
                    feature_points = {}
                    for bin_value, woe_value in binning.woe_dict.items():
                        points = self.factor * (feature_coef * woe_value)
                        feature_points[bin_value] = round(points)

                    self.scorecard_table[feature_name] = feature_points
                    self.feature_contributions[feature_name] = {
                        'coefficient': feature_coef,
                        'points_range': f"{min(feature_points.values())} to {max(feature_points.values())}"
                    }
                    total_features += 1

            logger.info(f"Scorecard fitted for {total_features} features")
            self.is_fitted = True

            return self

        except Exception as e:
            logger.error(f"Error fitting scorecard: {e}")
            raise

    def calculate_score(self, X: pd.DataFrame, feature_names: List[str]) -> np.array:
        """
        Calculate credit scores for given data

        Parameters:
        - X: Input features DataFrame
        - feature_names: Feature names

        Returns:
        - Array of credit scores
        """
        if not self.is_fitted:
            raise ValueError("Scorecard must be fitted before calculating scores")

        scores = np.full(len(X), self.scorecard_table['base_points'])

        for feature_name in feature_names:
            if feature_name in self.scorecard_table:
                feature_points = self.scorecard_table[feature_name]

                for idx, value in enumerate(X[feature_name]):
                    # Find which bin the value belongs to
                    assigned_points = 0
                    for bin_interval, points in feature_points.items():
                        if hasattr(bin_interval, 'left'):  # pd.Interval
                            if bin_interval.left <= value <= bin_interval.right:
                                assigned_points = points
                                break
                        else:  # categorical
                            if value == bin_interval:
                                assigned_points = points
                                break

                    scores[idx] += assigned_points

        return scores

    def get_scorecard_table(self) -> pd.DataFrame:
        """Get formatted scorecard table"""
        if not self.is_fitted:
            raise ValueError("Scorecard must be fitted first")

        rows = []

        # Base points
        rows.append({
            'Feature': 'Base Score',
            'Bin': 'N/A',
            'Points': self.scorecard_table['base_points']
        })

        # Feature points
        for feature_name, bins in self.scorecard_table.items():
            if feature_name == 'base_points':
                continue

            for bin_value, points in bins.items():
                rows.append({
                    'Feature': feature_name,
                    'Bin': str(bin_value),
                    'Points': points
                })

        return pd.DataFrame(rows)

    def get_scorecard_summary(self) -> Dict:
        """Get scorecard summary statistics"""
        if not self.is_fitted:
            raise ValueError("Scorecard must be fitted first")

        scorecard_df = self.get_scorecard_table()

        # Calculate score range
        total_points = scorecard_df.groupby('Feature')['Points'].agg(['min', 'max'])
        min_score = total_points['min'].sum()
        max_score = total_points['max'].sum()

        # Feature contributions
        feature_points_range = {}
        for feature in self.scorecard_table.keys():
            if feature != 'base_points':
                points = list(self.scorecard_table[feature].values())
                feature_points_range[feature] = {
                    'min_points': min(points),
                    'max_points': max(points),
                    'range': max(points) - min(points)
                }

        return {
            'score_range': {'min': min_score, 'max': max_score},
            'base_score': self.scorecard_table['base_points'],
            'num_features': len([k for k in self.scorecard_table.keys() if k != 'base_points']),
            'feature_contributions': feature_points_range,
            'pdo': self.pdo,
            'odds_at_base': self.odds
        }

    def plot_scorecard(self, save_path: str = None):
        """Create scorecard visualization"""
        if not self.is_fitted:
            raise ValueError("Scorecard must be fitted first")

        scorecard_df = self.get_scorecard_table()

        # Filter out base score for plotting
        feature_df = scorecard_df[scorecard_df['Feature'] != 'Base Score']

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Points distribution by feature
        features = feature_df['Feature'].unique()
        feature_points = []

        for feature in features:
            points = feature_df[feature_df['Feature'] == feature]['Points'].values
            feature_points.append({
                'feature': feature,
                'min_points': min(points),
                'max_points': max(points),
                'range': max(points) - min(points)
            })

        points_df = pd.DataFrame(feature_points)

        # Bar plot of points range
        ax1.bar(range(len(points_df)), points_df['range'], alpha=0.7, color='skyblue')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Points Range')
        ax1.set_title('Points Range by Feature')
        ax1.set_xticks(range(len(points_df)))
        ax1.set_xticklabels([f'Feature {i+1}' for i in range(len(points_df))], rotation=45)

        # Add value labels
        for i, row in points_df.iterrows():
            ax1.text(i, row['range'] + 0.5, f"{row['min_points']:.0f} to {row['max_points']:.0f}",
                    ha='center', va='bottom', fontsize=8)

        # Scorecard summary
        summary = self.get_scorecard_summary()
        summary_text = ".1f"".1f"f"""Scorecard Summary:
• Score Range: {summary['score_range']['min']:.0f} - {summary['score_range']['max']:.0f}
• Base Score: {summary['base_score']}
• Features: {summary['num_features']}
• PDO: {summary['pdo']} points
• Odds at Base: {summary['odds_at_base']}:1"""

        ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_title('Scorecard Summary')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Scorecard plot saved to {save_path}")
        else:
            plt.show()

        return fig

    def create_scorecard_report(self, output_path: str):
        """Create detailed scorecard report"""
        if not self.is_fitted:
            raise ValueError("Scorecard must be fitted first")

        with open(output_path, 'w') as f:
            f.write("CREDIT SCORECARD REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Summary
            summary = self.get_scorecard_summary()
            f.write("SCORECARD SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Score Range: {summary['score_range']['min']:.0f} - {summary['score_range']['max']:.0f}\n")
            f.write(f"Base Score: {summary['base_score']}\n")
            f.write(f"Number of Features: {summary['num_features']}\n")
            f.write(f"Points to Double Odds (PDO): {summary['pdo']}\n")
            f.write(f"Odds at Base Score: {summary['odds_at_base']}:1\n\n")

            # Feature contributions
            f.write("FEATURE CONTRIBUTIONS:\n")
            f.write("-" * 30 + "\n")
            for feature, contrib in summary['feature_contributions'].items():
                f.write(f"{feature}:\n")
                f.write(f"  Points Range: {contrib['min_points']} to {contrib['max_points']}\n")
                f.write(f"  Points Spread: {contrib['range']}\n\n")

            # Full scorecard table
            f.write("SCORECARD TABLE:\n")
            f.write("-" * 30 + "\n")
            scorecard_df = self.get_scorecard_table()
            f.write(scorecard_df.to_string(index=False))

        logger.info(f"Scorecard report saved to {output_path}")


def create_scorecard_visualization(scores: np.array, y_true: np.array,
                                 save_path: str = None, title: str = "Credit Score Distribution"):
    """
    Create visualization of credit score distribution

    Parameters:
    - scores: Array of credit scores
    - y_true: True labels
    - save_path: Path to save plot
    - title: Plot title
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Score distribution
    ax1.hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Credit Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Credit Score Distribution')
    ax1.grid(True, alpha=0.3)

    # Score by outcome
    good_scores = scores[y_true == 0]
    bad_scores = scores[y_true == 1]

    ax2.hist(good_scores, bins=20, alpha=0.7, label='Good', color='green')
    ax2.hist(bad_scores, bins=20, alpha=0.7, label='Bad', color='red')
    ax2.set_xlabel('Credit Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution by Outcome')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Score bands analysis
    score_bands = pd.cut(scores, bins=10, labels=[f'Band_{i+1}' for i in range(10)])
    band_analysis = pd.DataFrame({
        'score': scores,
        'band': score_bands,
        'target': y_true
    })

    band_stats = band_analysis.groupby('band').agg({
        'target': ['count', 'mean']
    }).round(4)
    band_stats.columns = ['count', 'bad_rate']

    # Bad rate by score band
    ax3.bar(range(len(band_stats)), band_stats['bad_rate'], alpha=0.7, color='red')
    ax3.set_xlabel('Score Band (Low to High)')
    ax3.set_ylabel('Bad Rate')
    ax3.set_title('Bad Rate by Score Band')
    ax3.set_xticks(range(len(band_stats)))
    ax3.set_xticklabels(band_stats.index, rotation=45)
    ax3.grid(True, alpha=0.3)

    # Cumulative bad rate
    sorted_indices = np.argsort(scores)[::-1]  # Sort by score descending
    sorted_targets = y_true[sorted_indices]
    cumulative_bad_rate = np.cumsum(sorted_targets) / np.arange(1, len(sorted_targets) + 1)

    ax4.plot(range(len(cumulative_bad_rate)), cumulative_bad_rate, color='purple')
    ax4.set_xlabel('Population (Ordered by Score)')
    ax4.set_ylabel('Cumulative Bad Rate')
    ax4.set_title('Cumulative Bad Rate (Ordered by Score)')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Scorecard visualization saved to {save_path}")
    else:
        plt.show()

    return fig


def generate_scorecard_report(scorecard: CreditScorecard, scores: np.array,
                            y_true: np.array, output_folder: str):
    """
    Generate comprehensive scorecard report with visualizations

    Parameters:
    - scorecard: Fitted CreditScorecard object
    - scores: Credit scores array
    - y_true: True labels
    - output_folder: Output folder path
    """
    os.makedirs(output_folder, exist_ok=True)

    # Scorecard table
    scorecard_table_path = os.path.join(output_folder, "scorecard_table.csv")
    scorecard_df = scorecard.get_scorecard_table()
    scorecard_df.to_csv(scorecard_table_path, index=False)

    # Scorecard report
    report_path = os.path.join(output_folder, "scorecard_report.txt")
    scorecard.create_scorecard_report(report_path)

    # Scorecard plot
    scorecard_plot_path = os.path.join(output_folder, "scorecard_analysis.png")
    scorecard.plot_scorecard(save_path=scorecard_plot_path)

    # Score distribution visualization
    score_viz_path = os.path.join(output_folder, "score_distribution_analysis.png")
    create_scorecard_visualization(scores, y_true, save_path=score_viz_path)

    # Scores DataFrame
    scores_df = pd.DataFrame({
        'credit_score': scores,
        'true_label': y_true
    })
    scores_path = os.path.join(output_folder, "credit_scores.csv")
    scores_df.to_csv(scores_path, index=False)

    logger.info(f"Comprehensive scorecard report generated in {output_folder}")


if __name__ == "__main__":
    # Example usage
    print("Credit Scorecard Utilities")
    print("Import this module to use CreditScorecard class")
