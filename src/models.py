from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
from enum import Enum

class FeeType(str, Enum):
    FIXED = "fixed"
    PERCENTAGE = "percentage"

class TargetSegment(str, Enum):
    ALL = "all"
    CRYPTO = "crypto"
    LOW_BALANCE = "low_balance"
    ACTIVE_TRADER = "active_trader"
    NEW_USER = "new_user"
    CUSTOM = "custom"

class FeatureSpecification(BaseModel):
    feature_name: str = Field(description="A short, descriptive name for the feature.")
    fee_type: FeeType = Field(description="The type of fee applied (fixed or percentage).")
    fee_value: float = Field(description="The numerical value of the fee (e.g., 2.0 for $2 or 0.5 for 0.5%).")
    applies_to: str = Field(description="Description of what the fee applies to (e.g., 'All withdrawals', 'crypto withdrawals').")
    condition: str = Field(description="A Pandas evaluation string condition to filter impacted customers (e.g., 'account_balance < 500', 'crypto_exposure == True', 'monthly_withdrawals > 5').")
    assumptions: List[str] = Field(description="A list of assumptions made by the parser.")

class RevenueScenarios(BaseModel):
    conservative: float
    realistic: float
    optimistic: float

class SimulationMetrics(BaseModel):
    total_revenue_estimate: RevenueScenarios
    impacted_customer_count: int
    impacted_customer_percentage: float
    income_distribution_impacted: Dict[str, float] = Field(description="Percentage distribution of impacted customers across income bands.")
    concentration_index: float = Field(description="Gini-like index measuring revenue concentration.")

class ValidationReport(BaseModel):
    is_valid: bool = Field(description="Whether the simulation metrics look correct and valid relative to the feature spec.")
    anomalies: List[str] = Field(description="List of identified data anomalies or logical inconsistencies, if any.")
    notes: str = Field(description="General notes on the simulation metrics validity.")

class RiskSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class RiskItem(BaseModel):
    title: str
    severity: RiskSeverity
    evidence: str
    mitigation: str

class LaunchRecommendation(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"

class RiskAnalysis(BaseModel):
    fairness_risks: List[RiskItem]
    regulatory_risks: List[RiskItem]
    reputational_risks: List[RiskItem]
    concentration_risks: List[RiskItem]
    overall_risk_level: RiskSeverity

class ComplianceChecklist(BaseModel):
    recommendation: LaunchRecommendation
    critical_actions: List[str]
    readiness_score: float = Field(description="Estimated readiness out of 100 based on the severity of the risks and actions needed.")
