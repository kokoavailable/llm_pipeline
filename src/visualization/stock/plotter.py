import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class StockVisualizer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def plot_stock_history(self, df, column='Adj Close', title='Stock Price History'):
        """주식 가격 히스토리를 시각화합니다."""
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.grid(True)
        plt.plot(df[column], label=column, color='b')
        plt.legend(loc='best')
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'stock_history.png'))
        
        plt.show()
    
    def plot_prediction_results(self, y_true, y_pred, window_size=None, features=None):
        """실제값과 예측값을 비교 시각화합니다."""
        plt.figure(figsize=(12, 6))
        
        if window_size and features:
            title = f"{', '.join(features)}, window_size={window_size}"
        else:
            title = "Prediction Results"
            
        plt.title(title)
        plt.ylabel('Price (normalized)')
        plt.xlabel('Time')
        plt.plot(y_true, label='Actual', color='b')
        plt.plot(y_pred, label='Prediction', color='r')
        plt.grid(True)
        plt.legend(loc='best')
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'prediction_results.png'))
        
        plt.show()
    
    def plot_training_history(self, history):
        """훈련 과정의 손실 값을 시각화합니다."""
        epochs = [entry['epoch'] for entry in history]
        train_losses = [entry['train_loss'] for entry in history]
        val_losses = [entry['val_loss'] for entry in history]
        
        plt.figure(figsize=(10, 6))
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.grid(True)
        plt.legend(loc='best')
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        
        plt.show()