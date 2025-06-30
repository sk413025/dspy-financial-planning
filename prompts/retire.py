import dspy
import datetime as dt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation import monte_carlo
import numpy as np


class Retire(dspy.Signature):
    """Retirement planning simulation signature."""
    
    yrs: int = dspy.InputField(desc="years until retirement")
    return_mu: float = dspy.InputField(desc="expected mean annual return (%)")
    return_sigma: float = dspy.InputField(desc="annual std-dev (%)")
    spend: float = dspy.InputField(desc="annual spend (TWD)")
    init_net: float = dspy.InputField(desc="current net worth (TWD)")
    inflation: float = dspy.InputField(desc="annual inflation (%)")
    goal_pct: float = dspy.InputField(desc="max bankruptcy probability (%)")


class RetireSim(dspy.Module):
    """Monte Carlo simulation module for retirement planning."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, **kwargs):
        """
        Run retirement simulation using monte_carlo.run_sim and return metrics.
        
        Parameters from kwargs:
        - yrs: years until retirement
        - return_mu: expected mean annual return (%)
        - return_sigma: annual std-dev (%)
        - spend: annual spend (TWD)
        - init_net: current net worth (TWD)
        - inflation: annual inflation (%)
        - goal_pct: max bankruptcy probability (%)
        
        Returns:
        - dict: Dictionary containing simulation metrics
        """
        # Extract parameters
        yrs = kwargs.get('yrs', 25)
        return_mu = kwargs.get('return_mu', 7.0) / 100.0  # Convert percentage to decimal
        return_sigma = kwargs.get('return_sigma', 15.0) / 100.0  # Convert percentage to decimal
        spend = kwargs.get('spend', 1000000.0)
        init_net = kwargs.get('init_net', 3000000.0)
        inflation = kwargs.get('inflation', 3.0) / 100.0  # Convert percentage to decimal
        goal_pct = kwargs.get('goal_pct', 5.0)
        
        # Run Monte Carlo simulation with updated signature
        results = monte_carlo.run_sim(
            mu=return_mu,
            sigma=return_sigma,
            yrs=yrs,
            init_net=init_net,
            spend=spend,
            inflation=inflation,
            n=10000
        )
        
        # Extract results
        paths = results['paths']
        final_balances = results['final_balance']
        bankruptcy_prob = results['bankruptcy_prob']
        
        # Calculate additional metrics
        positive_balances = final_balances[final_balances > 0]
        
        # Create metrics dictionary
        metrics = {
            'bankruptcy_probability': bankruptcy_prob,
            'meets_goal': bankruptcy_prob <= goal_pct,
            'median_end_balance': float(np.median(final_balances)),
            'mean_end_balance': float(np.mean(final_balances)),
            'percentile_10': float(np.percentile(final_balances, 10)),
            'percentile_90': float(np.percentile(final_balances, 90)),
            'mean_positive_balance': float(np.mean(positive_balances)) if len(positive_balances) > 0 else 0.0,
            'n_simulations': 10000
        }
        
        return metrics


if __name__ == "__main__":
    # Create the module
    module = RetireSim()
    
    # Run a test example
    print("Running RetireSim with test parameters...")
    test_params = {
        'yrs': 25,
        'return_mu': 7.0,
        'return_sigma': 15.0,
        'spend': 1000000.0,
        'init_net': 3000000.0,
        'inflation': 3.0,
        'goal_pct': 5.0
    }
    
    result = module.forward(**test_params)
    
    print("\nTest Results:")
    print(f"Bankruptcy Probability: {result['bankruptcy_probability']:.2f}%")
    print(f"Meets Goal: {result['meets_goal']}")
    print(f"Median End Balance: TWD {result['median_end_balance']:,.0f}")
    print(f"Mean End Balance: TWD {result['mean_end_balance']:,.0f}")
    print(f"10th Percentile: TWD {result['percentile_10']:,.0f}")
    print(f"90th Percentile: TWD {result['percentile_90']:,.0f}")
    print(f"\nModule compiled successfully!")