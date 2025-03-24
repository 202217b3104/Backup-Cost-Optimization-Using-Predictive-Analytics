# Treat this file as a module
from __future__ import annotations

from .data import CohesityDataProcessor
import pandas as pd
import numpy as np
from datetime import datetime

class BackupCostOptimizer:
    def __init__(self, csv_file=None):
        """
        Initialize the backup cost optimizer.
        If csv_file is provided, load the data; otherwise, set data later via set_data().
        """
        if csv_file:
            self.processor = CohesityDataProcessor(csv_file)
            self.processor.load_data()
            self.df = self.processor.df
        else:
            self.df = None
        # Default storage tier costs (per GB per month)
        self.storage_tiers = {
            'hot': {'cost_per_gb_month': 0.05, 'min_retention': 0},
            'warm': {'cost_per_gb_month': 0.02, 'min_retention': 30},
            'cold': {'cost_per_gb_month': 0.005, 'min_retention': 90},
            'archive': {'cost_per_gb_month': 0.001, 'min_retention': 365}
        }

    def set_data(self, df):
        """Set the data directly from a DataFrame."""
        self.df = df

    def analyze_backup_patterns(self):
        """
        Analyze backup jobs to identify patterns and optimization opportunities.
        Expects columns: date, job_id, job_name, backup_size_GB, storage_used_GB, dedup_ratio, cost_per_GB, retention_days.
        Returns a dictionary with job statistics and total storage aggregation.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please load data using set_data() or via csv_file.")
        
        analysis = {}
        # Group by job_name to get aggregated statistics
        job_stats = self.df.groupby('job_name').agg({
            'backup_size_GB': ['mean', 'sum', 'count'],
            'storage_used_GB': ['mean', 'sum']
        }).reset_index()
        # Compute deduplication efficiency per job
        job_stats['dedup_ratio'] = job_stats[('backup_size_GB', 'sum')] / job_stats[('storage_used_GB', 'sum')]
        # Flatten column names
        job_stats.columns = ['job_name', 'backup_size_mean_GB', 'backup_size_total_GB', 'backup_count',
                             'storage_used_mean_GB', 'storage_used_total_GB', 'dedup_ratio']
        
        analysis['job_stats'] = job_stats
        
        # Aggregate total storage by date
        total_storage = self.df.groupby('date').agg({
            'storage_used_GB': 'sum'
        }).reset_index()
        analysis['total_storage'] = total_storage
        
        return analysis

    def recommend_retention_policies(self, backup_analysis=None, legal_min_retention=30):
        """
        Recommend retention policies based on backup analysis.
        
        Args:
            backup_analysis (dict): Analysis results from analyze_backup_patterns.
            legal_min_retention (int): Minimum retention period required (days).
            
        Returns:
            DataFrame: Retention policy recommendations.
        """
        if backup_analysis is None:
            backup_analysis = self.analyze_backup_patterns()
        
        job_stats = backup_analysis['job_stats']
        recommendations = []
        
        for _, job in job_stats.iterrows():
            if job['backup_count'] < 5:
                continue  # Skip jobs with insufficient data
            
            job_name = job['job_name']
            # Get current retention; if not available, assume 90 days.
            current_retention = self.df[self.df['job_name'] == job_name]['retention_days'].mode().iloc[0] \
                                if not self.df[self.df['job_name'] == job_name]['retention_days'].empty else 90
            
            # Recommend retention based on dedup_ratio (a simplistic rule)
            if job['dedup_ratio'] < 1.5:
                suggested_retention = max(legal_min_retention, min(current_retention, 60))
                tier = 'hot' if current_retention <= 30 else 'warm'
            elif job['dedup_ratio'] < 3.0:
                suggested_retention = max(legal_min_retention, min(current_retention, 90))
                tier = 'warm'
            else:
                suggested_retention = max(legal_min_retention, current_retention)
                tier = 'cold' if current_retention >= 90 else 'warm'
            
            current_cost = self._calculate_storage_cost(job['storage_used_total_GB'] * (1024**3), self.storage_tiers['hot']['cost_per_gb_month'])
            suggested_cost = self._calculate_storage_cost(job['storage_used_total_GB'] * (suggested_retention / current_retention) * (1024**3), self.storage_tiers[tier]['cost_per_gb_month'])
            savings = current_cost - suggested_cost
            
            recommendations.append({
                'job_name': job_name,
                'current_retention_days': current_retention,
                'suggested_retention_days': suggested_retention,
                'suggested_storage_tier': tier,
                'current_monthly_cost_usd': current_cost,
                'suggested_monthly_cost_usd': suggested_cost,
                'monthly_savings_usd': savings,
                'annual_savings_usd': savings * 12,
                'confidence': 'high' if job['backup_count'] > 20 else 'medium'
            })
        return pd.DataFrame(recommendations)

    def recommend_storage_tiering(self):
        """
        Recommend storage tiering based on backup age.
        Returns a dictionary with tier recommendations, summary, and cost savings.
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        # Ensure the 'date' column is datetime
        if self.df['date'].dtype == 'object':
            self.df['date'] = pd.to_datetime(self.df['date'], format='%d-%m-%Y', errors='coerce')
        
        current_date = self.df['date'].max()
        df = self.df.copy()
        df['age_days'] = (current_date - df['date']).dt.days
        
        tier_assignments = []
        for _, row in df.iterrows():
            age = row['age_days']
            if age <= 30:
                tier = 'hot'
            elif age <= 90:
                tier = 'warm'
            else:
                tier = 'cold'
            tier_assignments.append({
                'job_id': row['job_id'],
                'job_name': row['job_name'],
                'date': row['date'],
                'age_days': age,
                'storage_used_GB': row['storage_used_GB'],
                'suggested_tier': tier
            })
        tier_recommendations = pd.DataFrame(tier_assignments)
        
        tier_summary = tier_recommendations.groupby('suggested_tier').agg({'storage_used_GB': 'sum'}).reset_index()
        tier_summary['monthly_cost_usd'] = tier_summary.apply(
            lambda x: x['storage_used_GB'] * self.storage_tiers.get(x['suggested_tier'], {}).get('cost_per_gb_month', 0.05),
            axis=1
        )
        current_cost = self._calculate_storage_cost(self.df['storage_used_GB'].sum() * (1024**3), self.storage_tiers['hot']['cost_per_gb_month'])
        optimized_cost = tier_summary['monthly_cost_usd'].sum()
        savings = current_cost - optimized_cost
        
        return {
            'tier_recommendations': tier_recommendations,
            'tier_summary': tier_summary,
            'current_monthly_cost_usd': current_cost,
            'optimized_monthly_cost_usd': optimized_cost,
            'monthly_savings_usd': savings,
            'annual_savings_usd': savings * 12
        }

    def recommend_deduplication_strategies(self, backup_analysis=None):
        """
        Recommend deduplication strategies based on backup analysis.
        Returns a DataFrame with deduplication recommendations.
        """
        if backup_analysis is None:
            backup_analysis = self.analyze_backup_patterns()
        job_stats = backup_analysis['job_stats']
        recommendations = []
        
        for _, job in job_stats.iterrows():
            if job['backup_count'] < 3:
                continue
            job_name = job['job_name']
            dedup_ratio = job['dedup_ratio']
            if dedup_ratio < 1.5:
                strategy = "Enable global deduplication"
                potential_savings = 0.3  # 30% improvement
                confidence = "medium"
            elif dedup_ratio < 3.0:
                strategy = "Optimize backup schedule for better deduplication"
                potential_savings = 0.15  # 15% improvement
                confidence = "medium"
            else:
                strategy = "Maintain current deduplication settings"
                potential_savings = 0.0
                confidence = "high"
            current_storage_bytes = job['storage_used_total_GB'] * (1024**3)
            potential_reduced_storage = current_storage_bytes * potential_savings
            cost_savings = self._calculate_storage_cost(potential_reduced_storage, self.storage_tiers['hot']['cost_per_gb_month'])
            
            recommendations.append({
                'job_name': job_name,
                'current_dedup_ratio': dedup_ratio,
                'recommended_strategy': strategy,
                'potential_storage_savings_bytes': potential_reduced_storage,
                'potential_monthly_cost_savings_usd': cost_savings,
                'potential_annual_cost_savings_usd': cost_savings * 12,
                'confidence': confidence
            })
        return pd.DataFrame(recommendations)

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive optimization report combining all analyses.
        Returns a dictionary with summary, backup analysis, retention, tiering, and deduplication recommendations.
        """
        backup_analysis = self.analyze_backup_patterns()
        retention_recs = self.recommend_retention_policies(backup_analysis)
        tiering_recs = self.recommend_storage_tiering()
        dedup_recs = self.recommend_deduplication_strategies(backup_analysis)
        
        retention_savings = retention_recs['monthly_savings_usd'].sum() if 'monthly_savings_usd' in retention_recs.columns else 0
        tiering_savings = tiering_recs.get('monthly_savings_usd', 0)
        dedup_savings = dedup_recs['potential_monthly_cost_savings_usd'].sum() if 'potential_monthly_cost_savings_usd' in dedup_recs.columns else 0
        total_monthly_savings = retention_savings + tiering_savings + dedup_savings
        total_annual_savings = total_monthly_savings * 12
        
        summary = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'total_jobs_analyzed': len(backup_analysis['job_stats']),
            'total_storage_GB': self.df['storage_used_GB'].sum(),
            'retention_policy_recommendations': len(retention_recs),
            'tiering_recommendations': len(tiering_recs['tier_recommendations']) if 'tier_recommendations' in tiering_recs else 0,
            'dedup_recommendations': len(dedup_recs),
            'estimated_monthly_savings_usd': total_monthly_savings,
            'estimated_annual_savings_usd': total_annual_savings,
            'savings_breakdown': {
                'retention_policy_savings': retention_savings * 12,
                'storage_tiering_savings': tiering_recs.get('annual_savings_usd', 0),
                'deduplication_savings': dedup_savings * 12
            }
        }
        
        return {
            'summary': summary,
            'backup_analysis': backup_analysis,
            'retention_recommendations': retention_recs,
            'tiering_recommendations': tiering_recs,
            'dedup_recommendations': dedup_recs
        }

    def _calculate_storage_cost(self, bytes_stored, cost_per_gb_month):
        """
        Calculate monthly storage cost in USD.
        
        Args:
            bytes_stored (float): Storage used in bytes.
            cost_per_gb_month (float): Cost per GB per month.
        
        Returns:
            float: Monthly cost in USD.
        """
        gb_stored = bytes_stored / (1024**3)
        return gb_stored * cost_per_gb_month

    def export_recommendations_to_csv(self, report, output_dir):
        """
        Export recommendations to CSV files.
        
        Args:
            report (dict): Comprehensive report from generate_comprehensive_report.
            output_dir (str): Directory to save CSV files.
        
        Returns:
            list: Paths to saved CSV files.
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        retention_path = os.path.join(output_dir, 'retention_recommendations.csv')
        report['retention_recommendations'].to_csv(retention_path, index=False)
        
        tiering_path = os.path.join(output_dir, 'tiering_recommendations.csv')
        report['tiering_recommendations']['tier_recommendations'].to_csv(tiering_path, index=False)
        
        dedup_path = os.path.join(output_dir, 'dedup_recommendations.csv')
        report['dedup_recommendations'].to_csv(dedup_path, index=False)
        
        summary_df = pd.DataFrame([report['summary']])
        summary_path = os.path.join(output_dir, 'optimization_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        return [retention_path, tiering_path, dedup_path, summary_path]

# Example usage:
if __name__ == "__main__":
    optimizer = BackupCostOptimizer("backup_data.csv")
    report = optimizer.generate_comprehensive_report()
    print(report['summary'])
