import pandas as pd
import numpy as np
from typing import Dict, Any
from src.models import FeatureSpecification, SimulationMetrics, RevenueScenarios

class FinancialSimulator:
    def __init__(self, data_path: str = "data/customers.csv"):
        self.df = pd.read_csv(data_path)
        
    def simulate(self, feature_spec: FeatureSpecification) -> SimulationMetrics:
        # Segment customers based on condition
        try:
            impacted_df = self.df.query(feature_spec.condition)
        except Exception as e:
            print(f"Warning: Failed to parse condition '{feature_spec.condition}'. Error: {str(e)}")
            print("Assuming all customers are impacted.")
            impacted_df = self.df.copy()
            
        impacted_count = len(impacted_df)
        total_customers = len(self.df)
        impacted_percentage = (impacted_count / total_customers) * 100 if total_customers > 0 else 0
        
        # Volume estimation & Revenue Calculation
        applies_target = feature_spec.applies_to.lower()
        
        # Heuristics for volume calculation
        if "withdrawal" in applies_target:
            base_volume = impacted_df['monthly_withdrawals'].sum()
            # assume each withdrawal is around $500 for % fees, or 1 unit for fixed
            volume_multiplier = 500 if feature_spec.fee_type == "percentage" else 1
        elif "fx" in applies_target or "foreign" in applies_target:
            base_volume = impacted_df['fx_transactions_mo'].sum()
            volume_multiplier = 1000 if feature_spec.fee_type == "percentage" else 1
        elif "maintenance" in applies_target or "account" in applies_target:
            base_volume = impacted_count
            volume_multiplier = 1 # generally fixed fee per account per month
        else:
            base_volume = impacted_count
            volume_multiplier = 1

        annual_events = base_volume * 12 # assuming monthly data
        
        if feature_spec.fee_type == "percentage":
            # fee_value is like 0.5 for 0.5%
            unit_revenue = volume_multiplier * (feature_spec.fee_value / 100)
        else:
            unit_revenue = feature_spec.fee_value
            
        max_annual_revenue = float(annual_events * unit_revenue)
        
        # Revenue scenarios
        # Conservative: 60% adoption/retention
        # Realistic: 85% adoption/retention
        # Optimistic: 100% adoption/retention
        revenue = RevenueScenarios(
            conservative=max_annual_revenue * 0.60,
            realistic=max_annual_revenue * 0.85,
            optimistic=max_annual_revenue * 1.00
        )
        
        # Fairness metrics: Income distribution impacted
        if impacted_count > 0:
            income_dist = impacted_df['income_band'].value_counts(normalize=True).to_dict()
            # Convert values to percentages
            income_dist = {k: float(v * 100) for k, v in income_dist.items()}
        else:
            income_dist = {}
            
        # Concentration index (Gini-like)
        concentration_index = self._calculate_gini(impacted_df['account_balance'].values)
        
        return SimulationMetrics(
            total_revenue_estimate=revenue,
            impacted_customer_count=impacted_count,
            impacted_customer_percentage=float(impacted_percentage),
            income_distribution_impacted=income_dist,
            concentration_index=float(concentration_index)
        )
        
    def _calculate_gini(self, array: np.ndarray) -> float:
        # Avoid zero division and negative values which break Gini
        if len(array) == 0:
            return 0.0
        array = np.sort(np.maximum(array, 0)) # Gini is well-defined for non-negative
        if np.sum(array) == 0:
            return 0.0
        index = np.arange(1, len(array) + 1)
        n = len(array)
        return float(((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))))
