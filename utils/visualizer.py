"""
Visualization tools for logged dspy experiments
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns
from utils.logger import get_logger


class ExperimentVisualizer:
    """Create visualizations from logged experiments"""
    
    def __init__(self, session_id: str = None):
        self.logger = get_logger()
        if session_id:
            self.session_file = Path("logs") / f"session_{session_id}.jsonl"
        else:
            self.session_file = self.logger.session_file
        
        self.entries = self._load_entries()
    
    def _load_entries(self) -> List[Dict]:
        """Load entries from JSONL file"""
        entries = []
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                for line in f:
                    entries.append(json.loads(line))
        return entries
    
    def plot_monte_carlo_paths(self, entry_id: int, save_path: str = None, n_paths: int = 50):
        """Plot Monte Carlo simulation paths for a specific entry"""
        entry = next((e for e in self.entries if e["id"] == entry_id), None)
        if not entry or "monte_carlo" not in entry["intermediate"]:
            print(f"No Monte Carlo data found for entry {entry_id}")
            return
        
        mc_data = entry["intermediate"]["monte_carlo"]
        if "sample_paths" not in mc_data:
            print(f"No sample paths saved for entry {entry_id}")
            return
        
        paths = np.array(mc_data["sample_paths"])
        years = range(len(paths[0]))
        
        plt.figure(figsize=(12, 8))
        
        # Plot sample paths
        for i in range(min(n_paths, len(paths))):
            plt.plot(years, paths[i], alpha=0.3, color='blue', linewidth=0.5)
        
        # Plot median path
        median_path = np.median(paths, axis=0)
        plt.plot(years, median_path, color='red', linewidth=2, label='Median Path')
        
        # Plot 10th and 90th percentiles
        p10 = np.percentile(paths, 10, axis=0)
        p90 = np.percentile(paths, 90, axis=0)
        plt.fill_between(years, p10, p90, alpha=0.2, color='green', label='10th-90th Percentile')
        
        plt.title(f'Monte Carlo Simulation Paths (Entry {entry_id})')
        plt.xlabel('Years')
        plt.ylabel('Net Worth (TWD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis to show values in millions
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_bankruptcy_distribution(self, save_path: str = None):
        """Plot distribution of bankruptcy probabilities across all entries"""
        bankruptcy_probs = []
        queries = []
        
        for entry in self.entries:
            if entry["output"] and "bankruptcy_probability" in entry["output"]:
                bankruptcy_probs.append(entry["output"]["bankruptcy_probability"])
                queries.append(f"Query {entry['id']}")
        
        if not bankruptcy_probs:
            print("No bankruptcy probability data found")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(bankruptcy_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(bankruptcy_probs), color='red', linestyle='--', label=f'Mean: {np.mean(bankruptcy_probs):.1f}%')
        plt.xlabel('Bankruptcy Probability (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Bankruptcy Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bar chart for individual queries
        plt.subplot(1, 2, 2)
        colors = ['red' if p > 50 else 'orange' if p > 20 else 'green' for p in bankruptcy_probs]
        plt.bar(range(len(bankruptcy_probs)), bankruptcy_probs, color=colors, alpha=0.7)
        plt.xlabel('Query')
        plt.ylabel('Bankruptcy Probability (%)')
        plt.title('Bankruptcy Probability by Query')
        plt.xticks(range(len(queries)), [f'Q{i+1}' for i in range(len(queries))], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_parameter_analysis(self, save_path: str = None):
        """Analyze how different parameters affect bankruptcy probability"""
        data = []
        
        for entry in self.entries:
            if (entry["intermediate"].get("parsing") and 
                entry["output"] and 
                "bankruptcy_probability" in entry["output"]):
                
                params = entry["intermediate"]["parsing"]["parsed_params"]
                bankruptcy_prob = entry["output"]["bankruptcy_probability"]
                
                data.append({
                    'yrs': params.get('yrs', 0),
                    'return_mu': params.get('return_mu', 0),
                    'return_sigma': params.get('return_sigma', 0),
                    'spend': params.get('spend', 0) / 1e6,  # Convert to millions
                    'init_net': params.get('init_net', 0) / 1e6,  # Convert to millions
                    'inflation': params.get('inflation', 0),
                    'bankruptcy_prob': bankruptcy_prob
                })
        
        if not data:
            print("No parameter data found")
            return
        
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        params = ['yrs', 'return_mu', 'return_sigma', 'spend', 'init_net', 'inflation']
        labels = ['Years to Retirement', 'Return (%)', 'Volatility (%)', 
                 'Annual Spend (M TWD)', 'Initial Net Worth (M TWD)', 'Inflation (%)']
        
        for i, (param, label) in enumerate(zip(params, labels)):
            axes[i].scatter(df[param], df['bankruptcy_prob'], alpha=0.7, s=50)
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('Bankruptcy Probability (%)')
            axes[i].set_title(f'Bankruptcy Prob vs {label}')
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line if there are enough points
            if len(df) > 1:
                z = np.polyfit(df[param], df['bankruptcy_prob'], 1)
                p = np.poly1d(z)
                axes[i].plot(df[param], p(df[param]), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def create_session_report(self, output_dir: str = None):
        """Create a comprehensive visual report of the session"""
        if output_dir is None:
            output_dir = Path("logs") / f"report_{self.logger.session_id}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"Generating visual report in {output_dir}")
        
        # 1. Plot bankruptcy distribution
        self.plot_bankruptcy_distribution(output_dir / "bankruptcy_distribution.png")
        
        # 2. Plot parameter analysis
        self.plot_parameter_analysis(output_dir / "parameter_analysis.png")
        
        # 3. Plot Monte Carlo paths for each entry
        for entry in self.entries:
            if "monte_carlo" in entry.get("intermediate", {}):
                self.plot_monte_carlo_paths(
                    entry["id"], 
                    output_dir / f"monte_carlo_paths_query_{entry['id']}.png"
                )
        
        # 4. Generate summary statistics plot
        self._plot_summary_stats(output_dir / "summary_stats.png")
        
        # 5. Generate text summary
        self.logger.generate_report(output_dir / "summary.md")
        
        print(f"Report generated successfully in {output_dir}")
        return output_dir
    
    def _plot_summary_stats(self, save_path: str):
        """Plot summary statistics of the session"""
        successful_queries = [e for e in self.entries if not e["errors"]]
        failed_queries = [e for e in self.entries if e["errors"]]
        
        # Processing times
        durations = [e["duration_ms"] for e in self.entries if e.get("duration_ms")]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rate pie chart
        axes[0, 0].pie([len(successful_queries), len(failed_queries)], 
                       labels=['Successful', 'Failed'], 
                       autopct='%1.1f%%',
                       colors=['green', 'red'])
        axes[0, 0].set_title('Query Success Rate')
        
        # Processing time distribution
        if durations:
            axes[0, 1].hist(durations, bins=10, alpha=0.7, color='skyblue')
            axes[0, 1].axvline(np.mean(durations), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(durations):.0f}ms')
            axes[0, 1].set_xlabel('Processing Time (ms)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Processing Time Distribution')
            axes[0, 1].legend()
        
        # Queries over time
        timestamps = [pd.to_datetime(e["timestamp"]) for e in self.entries]
        query_counts = list(range(1, len(timestamps) + 1))
        
        axes[1, 0].plot(timestamps, query_counts, marker='o')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Cumulative Queries')
        axes[1, 0].set_title('Queries Over Time')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Goal achievement rate
        meets_goal = [e["output"].get("meets_goal", False) for e in successful_queries if e.get("output")]
        if meets_goal:
            goal_success_rate = sum(meets_goal) / len(meets_goal) * 100
            axes[1, 1].bar(['Meets Goal', 'Does Not Meet Goal'], 
                          [sum(meets_goal), len(meets_goal) - sum(meets_goal)],
                          color=['green', 'orange'])
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title(f'Goal Achievement Rate: {goal_success_rate:.1f}%')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def generate_report_cli():
    """CLI function to generate reports"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visual report from logged experiments')
    parser.add_argument('--session-id', help='Specific session ID to analyze')
    parser.add_argument('--output-dir', help='Output directory for report')
    
    args = parser.parse_args()
    
    visualizer = ExperimentVisualizer(args.session_id)
    visualizer.create_session_report(args.output_dir)


if __name__ == "__main__":
    generate_report_cli()