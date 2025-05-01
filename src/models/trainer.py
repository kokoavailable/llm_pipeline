import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from torch.utils.data import DataLoader, TensorDataset
import time
import json

class ModelTrainer:
    def __init__(self, model, config_path):
        self.model = model
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.learning_rate = self.config["training"]["learning_rate"]
        self.batch_size = self.config["training"]["batch_size"]
        self.num_epochs = self.config["training"]["epochs"]
        self.patience = self.config["training"]["patience"]
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def prepare_data_loaders(self, x_train, y_train, x_test, y_test):
        """데이터 로더를 준비합니다."""
        # 텐서 변환
        x_train_tensor = torch.FloatTensor(x_train)
        y_train_tensor = torch.FloatTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # DataLoader 생성
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader, log_dir=None):
        """모델을 훈련시킵니다."""
        best_val_loss = float('inf')
        counter = 0
        training_history = []
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # 훈련
            self.model.train()
            train_losses = []
            
            for x_batch, y_batch in train_loader:
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # 검증
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for x_val, y_val in test_loader:
                    val_output = self.model(x_val)
                    val_loss = self.criterion(val_output, y_val)
                    val_losses.append(val_loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_elapsed": time.time() - start_time
            }
            
            training_history.append(epoch_info)
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        # 훈련 기록 저장
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "training_history.json"), "w") as f:
                json.dump(training_history, f)
        
        return self.model, training_history
    
    def predict(self, x_test_tensor):
        """테스트 데이터에 대한 예측을 수행합니다."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_test_tensor).numpy()
        return predictions