import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path

class StockDataLoader:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.ticker_symbol = self.config["model"]["ticker"]
        
    def load_data(self):
        """주식 데이터를 로드하고 이동평균을 계산합니다."""
        ticker = yf.Ticker(self.ticker_symbol)
        raw_df = ticker.history(period="max", auto_adjust=False)
        
        # 이동평균 계산
        raw_df['3MA'] = raw_df['Adj Close'].rolling(window=3).mean()
        raw_df['5MA'] = raw_df['Adj Close'].rolling(window=5).mean()
        
        return raw_df
    
    def save_raw_data(self, df, output_dir):
        """원시 데이터를 저장합니다."""
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"{self.ticker_symbol}_stock_data.csv"))
        return os.path.join(output_dir, f"{self.ticker_symbol}_stock_data.csv")