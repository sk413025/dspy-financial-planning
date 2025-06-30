import dspy
import json
import sys
import os

# Add parent directory to path to import from prompts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.retire import RetireSim
from utils.logger import get_logger


class QueryParser(dspy.Signature):
    """Parse natural language retirement query into structured parameters."""
    
    query: str = dspy.InputField(desc="Natural language retirement planning query")
    yrs: int = dspy.OutputField(desc="Years until retirement")
    return_mu: float = dspy.OutputField(desc="Expected annual return as percentage (e.g., 7.0 for 7%)")
    return_sigma: float = dspy.OutputField(desc="Annual volatility as percentage (e.g., 15.0 for 15%)")
    spend: float = dspy.OutputField(desc="Annual spending in TWD")
    init_net: float = dspy.OutputField(desc="Current net worth in TWD")
    inflation: float = dspy.OutputField(desc="Annual inflation as percentage (e.g., 3.0 for 3%)")
    goal_pct: float = dspy.OutputField(desc="Maximum acceptable bankruptcy probability as percentage")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py \"<natural language query>\"")
        print("Example: python run.py \"If I retire in 25 years with 7% return and 15% volatility, spending 1M TWD annually with 3M TWD saved, what's my bankruptcy risk?\"")
        sys.exit(1)
    
    # Get the natural language query from command line
    nl_query = " ".join(sys.argv[1:])
    
    # Initialize logger
    logger = get_logger()
    logger.start_query(nl_query, source="cli")
    
    try:
        # Initialize dspy with a language model
        # Configure with OpenAI GPT model
        import os
        
        # Setup API key from config
        try:
            from config import OPENAI_API_KEY
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        except ImportError:
            error_msg = "config.py not found"
            logger.log_error(error_msg, "initialization")
            logger.save_entry()
            print("Error: config.py not found")
            print("Please copy config.example.py to config.py and add your API key")
            sys.exit(1)
        
        # Configure dspy with OpenAI
        # Use LM class with provider
        lm = dspy.LM(model='openai/gpt-4o-mini', max_tokens=500)
        dspy.configure(lm=lm)
        
        # Create parser agent
        parser = dspy.ChainOfThought(QueryParser)
        
        # Parse the natural language query
        print(f"Parsing query: {nl_query}")
        parsed = parser(query=nl_query)
        
        # Extract parameters with defaults for missing values
        params = {
            'yrs': parsed.yrs if hasattr(parsed, 'yrs') else 25,
            'return_mu': parsed.return_mu if hasattr(parsed, 'return_mu') else 7.0,
            'return_sigma': parsed.return_sigma if hasattr(parsed, 'return_sigma') else 15.0,
            'spend': parsed.spend if hasattr(parsed, 'spend') else 1000000.0,
            'init_net': parsed.init_net if hasattr(parsed, 'init_net') else 3000000.0,
            'inflation': parsed.inflation if hasattr(parsed, 'inflation') else 3.0,
            'goal_pct': parsed.goal_pct if hasattr(parsed, 'goal_pct') else 5.0
        }
        
        # Log parsing results
        logger.log_parsing(params, str(parsed))
        
        print(f"\nParsed parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Initialize and run the retirement simulation
        sim = RetireSim()
        
        # Convert percentages to decimals for the simulation
        sim_params = params.copy()
        sim_params['return_mu'] = params['return_mu'] / 100.0
        sim_params['return_sigma'] = params['return_sigma'] / 100.0
        sim_params['inflation'] = params['inflation'] / 100.0
        
        # Log Monte Carlo start
        logger.log_monte_carlo_start(sim_params)
        
        # Run simulation
        print("\nRunning Monte Carlo simulation...")
        from simulation.monte_carlo import run_sim
        
        results = run_sim(
            mu=sim_params['return_mu'],
            sigma=sim_params['return_sigma'],
            yrs=params['yrs'],
            init_net=params['init_net'],
            spend=params['spend'],
            inflation=sim_params['inflation'],
            n=10000
        )
        
        # Log Monte Carlo results
        logger.log_monte_carlo_results(results)
        
        # Prepare output metrics
        import numpy as np
        positive_balances = results['final_balance'][results['final_balance'] > 0]
        
        metrics = {
            'bankruptcy_probability': float(results['bankruptcy_prob']),
            'meets_goal': bool(results['bankruptcy_prob'] <= params['goal_pct']),
            'final_balance_mean': float(np.mean(results['final_balance'])),
            'final_balance_median': float(np.median(results['final_balance'])),
            'final_balance_positive_mean': float(np.mean(positive_balances)) if len(positive_balances) > 0 else 0.0,
            'final_balance_10th_percentile': float(np.percentile(results['final_balance'], 10)),
            'final_balance_90th_percentile': float(np.percentile(results['final_balance'], 90)),
            'parameters': params
        }
        
        # Log final output
        logger.log_final_output(metrics)
        
        # Print JSON output
        print("\nResults:")
        print(json.dumps(metrics, indent=2))
        
        # Save log entry
        logger.save_entry()
        
        print(f"\n📊 Session logged to: {logger.session_file}")
        print(f"📝 Summary available at: {logger.summary_file}")
        
    except Exception as e:
        # Log error
        logger.log_error(str(e), "execution")
        logger.save_entry()
        
        error_output = {
            'error': str(e),
            'query': nl_query
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()