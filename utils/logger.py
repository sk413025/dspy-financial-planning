"""
Logging system for tracking dspy inputs, outputs, and intermediate results
"""
import json
import datetime
import os
from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path


class ExperimentLogger:
    """Logger for tracking all dspy experiments and results"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session ID
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"session_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"summary_{self.session_id}.json"
        
        # Initialize tracking
        self.entries = []
        self.current_entry = None
        
    def start_query(self, query: str, source: str = "cli"):
        """Start tracking a new query"""
        self.current_entry = {
            "id": len(self.entries) + 1,
            "timestamp": datetime.datetime.now().isoformat(),
            "source": source,
            "input": {
                "raw_query": query
            },
            "intermediate": {},
            "output": {},
            "errors": [],
            "duration_ms": None
        }
        self.start_time = datetime.datetime.now()
        
    def log_parsing(self, parsed_params: Dict[str, Any], raw_response: Optional[str] = None):
        """Log the parsing phase results"""
        if self.current_entry:
            self.current_entry["intermediate"]["parsing"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "parsed_params": parsed_params,
                "raw_llm_response": raw_response
            }
    
    def log_monte_carlo_start(self, params: Dict[str, Any]):
        """Log Monte Carlo simulation parameters"""
        if self.current_entry:
            self.current_entry["intermediate"]["monte_carlo"] = {
                "start_timestamp": datetime.datetime.now().isoformat(),
                "input_params": params,
                "paths": None,  # Will be updated
                "results": None
            }
    
    def log_monte_carlo_results(self, results: Dict[str, Any]):
        """Log Monte Carlo simulation results"""
        if self.current_entry:
            mc_data = self.current_entry["intermediate"].get("monte_carlo", {})
            mc_data.update({
                "end_timestamp": datetime.datetime.now().isoformat(),
                "results": {
                    "bankruptcy_prob": float(results["bankruptcy_prob"]),
                    "final_balance_stats": {
                        "mean": float(results["final_balance"].mean()),
                        "median": float(results["final_balance"][results["final_balance"] > 0].mean()) if any(results["final_balance"] > 0) else 0,
                        "std": float(results["final_balance"].std()),
                        "min": float(results["final_balance"].min()),
                        "max": float(results["final_balance"].max())
                    },
                    "n_simulations": len(results["final_balance"])
                }
            })
            
            # Save sample paths for visualization (first 100)
            import numpy as np
            sample_paths = results["paths"][:100].tolist() if len(results["paths"]) > 100 else results["paths"].tolist()
            mc_data["sample_paths"] = sample_paths
            
            self.current_entry["intermediate"]["monte_carlo"] = mc_data
    
    def log_final_output(self, metrics: Dict[str, Any]):
        """Log the final output metrics"""
        if self.current_entry:
            self.current_entry["output"] = metrics
            
            # Calculate duration
            duration = (datetime.datetime.now() - self.start_time).total_seconds() * 1000
            self.current_entry["duration_ms"] = duration
    
    def log_error(self, error: str, phase: str = "unknown"):
        """Log any errors that occur"""
        if self.current_entry:
            self.current_entry["errors"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "phase": phase,
                "error": str(error)
            })
    
    def log_info(self, message: str, phase: str = "general"):
        """Log general information"""
        if self.current_entry:
            if "info" not in self.current_entry["intermediate"]:
                self.current_entry["intermediate"]["info"] = []
            self.current_entry["intermediate"]["info"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "phase": phase,
                "message": message
            })
    
    def log_step(self, step_info: Dict[str, Any]):
        """Log a step in the demonstration process"""
        if self.current_entry:
            if "steps" not in self.current_entry["intermediate"]:
                self.current_entry["intermediate"]["steps"] = []
            step_info["timestamp"] = datetime.datetime.now().isoformat()
            self.current_entry["intermediate"]["steps"].append(step_info)
    
    def log_parameters(self, params: Dict[str, Any], step: str = "general"):
        """Log parameters for a specific step"""
        if self.current_entry:
            if "parameters" not in self.current_entry["intermediate"]:
                self.current_entry["intermediate"]["parameters"] = {}
            self.current_entry["intermediate"]["parameters"][step] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "params": params
            }
    
    def log_prediction_result(self, result_data: Dict[str, Any]):
        """Log the result of a prediction"""
        if self.current_entry:
            if "predictions" not in self.current_entry["intermediate"]:
                self.current_entry["intermediate"]["predictions"] = []
            result_data["timestamp"] = datetime.datetime.now().isoformat()
            self.current_entry["intermediate"]["predictions"].append(result_data)
    
    def log_training_examples(self, examples: list):
        """Log training examples used for Few-Shot learning"""
        if self.current_entry:
            self.current_entry["intermediate"]["training_examples"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "count": len(examples),
                "examples": examples
            }
    
    def log_comparison_results(self, comparison_data: Dict[str, Any]):
        """Log comparison results between different methods"""
        if self.current_entry:
            comparison_data["timestamp"] = datetime.datetime.now().isoformat()
            self.current_entry["intermediate"]["comparison"] = comparison_data
    
    def log_demo_summary(self, summary_data: Dict[str, Any]):
        """Log final summary of the demonstration"""
        if self.current_entry:
            summary_data["timestamp"] = datetime.datetime.now().isoformat()
            self.current_entry["output"]["demo_summary"] = summary_data
    
    def save_entry(self):
        """Save the current entry to file"""
        if self.current_entry:
            # Append to JSONL file
            with open(self.session_file, "a") as f:
                f.write(json.dumps(self.current_entry) + "\n")
            
            # Add to entries list
            self.entries.append(self.current_entry)
            
            # Update summary
            self._update_summary()
            
            # Reset current entry
            self.current_entry = None
    
    def _update_summary(self):
        """Update the summary file with session statistics"""
        summary = {
            "session_id": self.session_id,
            "total_queries": len(self.entries),
            "successful_queries": len([e for e in self.entries if not e["errors"]]),
            "failed_queries": len([e for e in self.entries if e["errors"]]),
            "average_duration_ms": sum(e["duration_ms"] for e in self.entries if e["duration_ms"]) / len(self.entries) if self.entries else 0,
            "queries": [
                {
                    "id": e["id"],
                    "timestamp": e["timestamp"],
                    "query": e["input"]["raw_query"],
                    "bankruptcy_prob": e["output"].get("bankruptcy_probability", None),
                    "success": len(e["errors"]) == 0
                }
                for e in self.entries
            ]
        }
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert logged entries to pandas DataFrame for analysis"""
        if not self.entries:
            return pd.DataFrame()
        
        rows = []
        for entry in self.entries:
            row = {
                "id": entry["id"],
                "timestamp": entry["timestamp"],
                "query": entry["input"]["raw_query"],
                "duration_ms": entry["duration_ms"],
                "success": len(entry["errors"]) == 0
            }
            
            # Add parsed parameters
            if "parsing" in entry["intermediate"]:
                params = entry["intermediate"]["parsing"]["parsed_params"]
                for key, value in params.items():
                    row[f"param_{key}"] = value
            
            # Add results
            if entry["output"]:
                row.update({
                    "bankruptcy_prob": entry["output"].get("bankruptcy_probability"),
                    "meets_goal": entry["output"].get("meets_goal"),
                    "final_balance_mean": entry["output"].get("final_balance_mean")
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_report(self, output_file: Optional[str] = None):
        """Generate a comprehensive report of the session"""
        if output_file is None:
            output_file = self.log_dir / f"report_{self.session_id}.md"
        
        df = self.get_dataframe()
        
        # Calculate average processing time safely
        avg_time = "N/A"
        if not df.empty and 'duration_ms' in df.columns:
            avg_time = f"{df['duration_ms'].mean():.0f}ms"
        
        report = f"""# Retirement Planning Session Report
Session ID: {self.session_id}
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics
- Total Queries: {len(self.entries)}
- Successful: {len([e for e in self.entries if not e["errors"]])}
- Failed: {len([e for e in self.entries if e["errors"]])}
- Average Processing Time: {avg_time}

## Query Details
"""
        
        for entry in self.entries:
            report += f"\n### Query {entry['id']}: {entry['input']['raw_query']}\n"
            report += f"- Timestamp: {entry['timestamp']}\n"
            report += f"- Duration: {entry['duration_ms']:.0f}ms\n"
            
            if "parsing" in entry["intermediate"]:
                params = entry["intermediate"]["parsing"]["parsed_params"]
                report += f"\n**Parsed Parameters:**\n"
                for key, value in params.items():
                    report += f"- {key}: {value}\n"
            
            if entry["output"]:
                report += f"\n**Results:**\n"
                report += f"- Bankruptcy Probability: {entry['output']['bankruptcy_probability']:.2f}%\n"
                report += f"- Meets Goal: {'Yes' if entry['output']['meets_goal'] else 'No'}\n"
                report += f"- Mean End Balance: TWD {entry['output']['final_balance_mean']:,.0f}\n"
            
            if entry["errors"]:
                report += f"\n**Errors:**\n"
                for error in entry["errors"]:
                    report += f"- [{error['phase']}] {error['error']}\n"
        
        # Save report
        with open(output_file, "w") as f:
            f.write(report)
        
        return report


# Global logger instance
_logger = None

def get_logger() -> ExperimentLogger:
    """Get or create the global logger instance"""
    global _logger
    if _logger is None:
        _logger = ExperimentLogger()
    return _logger