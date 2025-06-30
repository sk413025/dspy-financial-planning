from pathlib import Path
import pandas as pd
import yaml
from typing import Union


class Ledger:
    def __init__(self, root: Path):
        """Initialize Ledger with root directory path.
        
        Args:
            root: Path object pointing to the root directory containing financial data files
        """
        self.root = root
        self.assets = None
        self.liabilities = None
        self.cashflow = None
    
    def load_assets(self):
        """Load assets data from CSV file."""
        assets_path = self.root / "assets.csv"
        if assets_path.exists():
            self.assets = pd.read_csv(assets_path)
            return self.assets
        else:
            raise FileNotFoundError(f"Assets file not found at {assets_path}")
    
    def load_liabilities(self):
        """Load liabilities data from CSV file."""
        liabilities_path = self.root / "liabilities.csv"
        if liabilities_path.exists():
            self.liabilities = pd.read_csv(liabilities_path)
            return self.liabilities
        else:
            raise FileNotFoundError(f"Liabilities file not found at {liabilities_path}")
    
    def load_cashflow(self):
        """Load cashflow data from YAML file."""
        cashflow_path = self.root / "cashflow.yaml"
        if cashflow_path.exists():
            with open(cashflow_path, 'r') as f:
                self.cashflow = yaml.safe_load(f)
            return self.cashflow
        else:
            raise FileNotFoundError(f"Cashflow file not found at {cashflow_path}")
    
    def net_worth(self, on_date: str) -> float:
        """Calculate net worth on a specific date.
        
        Args:
            on_date: Date string for which to calculate net worth
            
        Returns:
            float: Net worth value (assets - liabilities) on the specified date
        """
        # Ensure data is loaded
        if self.assets is None:
            self.load_assets()
        if self.liabilities is None:
            self.load_liabilities()
        
        # Convert date columns to datetime if needed
        if 'date' in self.assets.columns:
            self.assets['date'] = pd.to_datetime(self.assets['date'])
        if 'date' in self.liabilities.columns:
            self.liabilities['date'] = pd.to_datetime(self.liabilities['date'])
        
        # Convert on_date to datetime
        target_date = pd.to_datetime(on_date)
        
        # Calculate total assets on the given date
        if 'date' in self.assets.columns:
            assets_on_date = self.assets[self.assets['date'] <= target_date]
            total_assets = assets_on_date['value'].sum() if 'value' in assets_on_date.columns else 0.0
        else:
            # If no date column, assume all assets are current
            total_assets = self.assets['value'].sum() if 'value' in self.assets.columns else 0.0
        
        # Calculate total liabilities on the given date
        if 'date' in self.liabilities.columns:
            liabilities_on_date = self.liabilities[self.liabilities['date'] <= target_date]
            total_liabilities = liabilities_on_date['value'].sum() if 'value' in liabilities_on_date.columns else 0.0
        else:
            # If no date column, assume all liabilities are current
            total_liabilities = self.liabilities['value'].sum() if 'value' in self.liabilities.columns else 0.0
        
        # Calculate net worth
        net_worth = total_assets - total_liabilities
        
        return float(net_worth)