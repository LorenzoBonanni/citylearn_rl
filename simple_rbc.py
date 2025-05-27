from citylearn.agents.rbc import OptimizedRBC
from citylearn.citylearn import CityLearnEnv
import numpy as np
import pandas as pd
import os
import json
import pathlib

class SimpleRBC(OptimizedRBC):
    """
    A simple Rule-Based Controller (RBC) for CityLearn that extends OptimizedRBC.
    Allows for testing different control strategies.
    """
    
def test_rbc_mode(schema_path="citylearn_challenge_2023_phase_2_local_evaluation", mode="normal"):
    """
    Test the SimpleRBC with a specific mode and return the evaluation results
    
    Parameters:
    -----------
    schema_path: str
        Path to the CityLearn environment schema
    mode: str
        Control mode for the SimpleRBC
        
    Returns:
    --------
    env: CityLearnEnv
        The environment after simulation
    kpis: pd.DataFrame
        Evaluation results dataframe
    """
    # Initialize environment
    env = CityLearnEnv(schema_path, central_agent=True)
    
    # Initialize SimpleRBC with the specified mode
    rbc = SimpleRBC(env, mode=mode)
    
    # Run the simulation
    observations, _ = env.reset()
    
    while not env.terminated:
        actions = rbc.predict(observations)
        observations, reward, info, terminated, truncated = env.step(actions)
    
    # Evaluate and return results
    kpis = env.evaluate()
    return env, kpis


if __name__ == "__main__":
    # Test different RBC modes
    modes = ["normal", "aggressive", "conservative", "night_charging"]
    results = {}
    
    # Create output directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "results", "csv")
    
    # Create directories if they don't exist
    pathlib.Path(csv_dir).mkdir(parents=True, exist_ok=True)
    
    for mode in modes:
        print(f"Testing {mode} mode...")
        env, kpis = test_rbc_mode(mode=mode)
        
        # Store pivoted results for comparison
        pivoted_kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
        results[mode] = pivoted_kpis
        
        # Print brief summary
        district_results = pivoted_kpis.get('District', pd.Series())
        print(f"\nResults for {mode} mode (District level):")
        for metric in ['electricity_consumption_total', 'cost_total', 'carbon_emissions_total', 'discomfort_proportion']:
            if metric in district_results:
                print(f"  {metric}: {district_results.get(metric, 'N/A')}")
        print("\n" + "-"*50)
    
    # Save results for later visualization
    for mode, result in results.items():
        result.to_csv(os.path.join(csv_dir, f"results_{mode}.csv"))
    
    print(f"\nTesting completed! Results saved in {csv_dir}")