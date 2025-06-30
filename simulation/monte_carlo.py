import numpy as np


def run_sim(mu: float, sigma: float, yrs: int, init_net: float, spend: float, 
            inflation: float, n: int = 10000) -> dict:
    """
    Run Monte Carlo simulation for retirement planning with inflation-adjusted spending.
    
    Parameters:
    -----------
    mu : float
        Expected annual return (e.g., 0.07 for 7%)
    sigma : float
        Annual volatility/standard deviation (e.g., 0.15 for 15%)
    yrs : int
        Number of years to simulate
    init_net : float
        Initial net worth
    spend : float
        Annual spending amount (in today's dollars)
    inflation : float
        Annual inflation rate (e.g., 0.03 for 3%)
    n : int, default=10000
        Number of simulation runs
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "paths": array of shape (n, yrs+1) with yearly balances for each simulation
        - "final_balance": array of shape (n,) with final balances
        - "bankruptcy_prob": probability of running out of money (as percentage)
    """
    # Initialize paths array to store yearly balances
    # Shape: (n simulations, yrs+1 time points including initial)
    paths = np.zeros((n, yrs + 1))
    paths[:, 0] = init_net  # Set initial balance
    
    # Generate random returns for all years and simulations at once
    # Using 365-day compounding: annual return = (1 + daily_return)^365 - 1
    # For log-normal with 365-day compounding, we adjust parameters
    daily_mu = mu / 365
    daily_sigma = sigma / np.sqrt(365)
    
    # Generate daily log returns and compound to annual
    daily_log_returns = np.random.normal(
        loc=daily_mu - 0.5 * daily_sigma**2,
        scale=daily_sigma,
        size=(n, yrs, 365)
    )
    
    # Convert to daily returns and compound to annual
    daily_returns = np.exp(daily_log_returns)
    annual_returns = np.prod(daily_returns, axis=2)
    
    # Simulate year by year
    for year in range(yrs):
        # Apply investment returns
        paths[:, year + 1] = paths[:, year] * annual_returns[:, year]
        
        # Subtract inflation-adjusted spending
        inflation_adjusted_spend = spend * ((1 + inflation) ** year)
        paths[:, year + 1] -= inflation_adjusted_spend
        
        # Prevent negative balances from growing (bankruptcy)
        paths[:, year + 1] = np.maximum(paths[:, year + 1], 0)
    
    # Extract final balances
    final_balance = paths[:, -1]
    
    # Calculate bankruptcy probability
    bankruptcy_count = np.sum(final_balance <= 0)
    bankruptcy_prob = (bankruptcy_count / n) * 100
    
    return {
        "paths": paths,
        "final_balance": final_balance,
        "bankruptcy_prob": bankruptcy_prob
    }