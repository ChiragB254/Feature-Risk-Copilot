import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_customers(num_customers=5000):
    np.random.seed(42)  # For reproducibility
    
    # Account balance: Log-normal (mean=$15k, heavy tail)
    mu = 8.5
    sigma = 1.5
    account_balance = np.random.lognormal(mean=mu, sigma=sigma, size=num_customers)
    account_balance = np.round(account_balance, 2)
    
    # Monthly withdrawals: Exponential (skew toward small)
    monthly_withdrawals = np.random.exponential(scale=2.0, size=num_customers).astype(int)
    
    # Crypto exposure: 35% True, 65% False
    crypto_exposure = np.random.choice([True, False], size=num_customers, p=[0.35, 0.65])
    
    # FX transactions / month: Poisson(lambda=2)
    fx_transactions_mo = np.random.poisson(lam=2, size=num_customers)
    
    # Income band: 50% low, 35% medium, 15% high
    income_band = np.random.choice(['Low (<$50k)', 'Medium ($50k-$100k)', 'High (>$100k)'], 
                                   size=num_customers, p=[0.50, 0.35, 0.15])
    
    # Is active trader: 30% True, 70% False
    is_active_trader = np.random.choice([True, False], size=num_customers, p=[0.30, 0.70])
    
    # Account age (months): Uniform(1, 120)
    account_age_months = np.random.randint(1, 121, size=num_customers)
    
    df = pd.DataFrame({
        'customer_id': np.arange(1, num_customers + 1),
        'account_balance': account_balance,
        'monthly_withdrawals': monthly_withdrawals,
        'crypto_exposure': crypto_exposure,
        'fx_transactions_mo': fx_transactions_mo,
        'income_band': income_band,
        'is_active_trader': is_active_trader,
        'account_age_months': account_age_months
    })
    
    return df

def main():
    df = generate_customers()
    
    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / 'customers.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} customers and saved to {output_path}")

if __name__ == "__main__":
    main()
